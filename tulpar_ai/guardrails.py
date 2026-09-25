"""Deterministic input and output guardrails. No LLM here: every decision is a regex over normalised text,
so it is fast, free, testable and cannot itself be prompt-injected.

Input (`check_input`) — one category per message, by precedence:
  self_harm        → escalate (a human must see it, even when the text is also rude)
  injection        → refuse
  pii_exfil        → refuse (asks for someone else's contacts or personal data)
  dangerous_domain → escalate (steroids, prescription drug dosing, starvation, extreme deficits)
  toxic            → calm boundary reply; skipped when the caller saw a pain/red-flag marker,
                     because a client in pain who swears still needs a human, not a lecture.
A HARD symptom (chest pain, fainting, blood) also drops injection and pii_exfil: the message goes to the trainer.

Output (`guard_reply`) — applied once in the service layer to every reply: masks contacts and card numbers,
blocks leaked prompt text, replaces medication dosage instructions (a dose of a nutrient such as calcium or caffeine
is not one).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from .pii import _EMAIL, _IIN, _PHONE

# ── normalisation ────────────────────────────────────────────────────────────
# Obfuscation tricks: latin look-alikes (xyй, cyka), digits (6ля), repeated letters (сууука),
# separators between letters (х.у.й, б-л-я), invisible characters (zero-width space, soft hyphen),
# fullwidth letters (ｆｕｃｋ). Two folds: to Cyrillic for ru/kk roots, to Latin for en roots.
_TO_CYR = str.maketrans({"a": "а", "b": "б", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м", "o": "о", "p": "р",
                         "t": "т", "x": "х", "y": "у", "0": "о", "3": "з", "4": "ч", "6": "б", "@": "а", "$": "с"})
_TO_LAT = str.maketrans({"а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y", "к": "k", "м": "m",
                         "т": "t", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_HAS_LETTER = re.compile(r"[a-zа-яәіңғүұқөһ]")
_HAS_CYR = re.compile(r"[а-яәіңғүұқөһ]")
# Latin letters that look like Cyrillic ones. A pure-Latin word is folded to Cyrillic only when it is built from
# these alone («cyka», «xyй» typed on the wrong layout); «debate» or «baseball» stay English.
_LOOKALIKE = set("abcehkmoptxy0346@$")
_SPACED = re.compile(r"(?<![\w*])[a-zа-я](?:[\s.\-_,]+[a-zа-я](?![\w*])){2,}")
_REPEAT = re.compile(r"([a-zа-яәіңғүұқөһ])\1+")


def _clean(text: str) -> str:
    """NFKC folds fullwidth and other compatibility forms to plain letters; format characters (zero-width space,
    soft hyphen, joiners, bidi marks) and stray combining accents are dropped, so one invisible character
    can no longer split a word. NFKC composes «й»/«ё» first, so their marks survive."""
    t = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in t if unicodedata.category(ch) not in ("Cf", "Mn"))


def _base(text: str) -> str:
    return re.sub(r"\s+", " ", _clean(text).lower().replace("ё", "е")).strip()


def _join_spaced(text: str) -> str:
    """«и г н о р и р у й», «х.у.й» → one word."""
    return _SPACED.sub(lambda m: re.sub(r"[\s.\-_,]+", "", m.group(0)), text)


def _to_cyr(tok: str) -> bool:
    if not _HAS_LETTER.search(tok):  # a plain «200» must stay a number
        return False
    return bool(_HAS_CYR.search(tok)) or set(re.sub(r"[^a-z0-9@$]", "", tok)) <= _LOOKALIKE


def _fold(text: str, table: dict) -> str:
    cyr = table is _TO_CYR
    toks = [t.translate(table) if (_to_cyr(t) if cyr else _HAS_LETTER.search(t)) else t for t in text.split(" ")]
    return _REPEAT.sub(r"\1", _join_spaced(" ".join(toks)))


def _fold_mixed(text: str) -> str:
    """Only mixed-script words get folded to Cyrillic («игн0рируй»), so English stays readable for the en patterns."""
    toks = []
    for t in _base(text).split(" "):
        if re.search(r"[а-я]", t) and re.search(r"[a-z0-9@$]", t):
            t = t.translate(_TO_CYR)
        toks.append(t)
    return " ".join(toks)


def normalize(text: str) -> tuple[str, str, str]:
    """(plain lower text, Cyrillic-folded, Latin-folded)."""
    low = _base(text)
    return low, _fold(low, _TO_CYR), _fold(low, _TO_LAT)


def _starred(*roots: str) -> str:
    """«х*й», «п**да», «f*ck»: the first letter is real, any later one may be masked by «*»."""
    return "|".join(rf"(?<![\w*]){r[0]}" + "".join(rf"(?:{c}|\*)" for c in r[1:]) for r in roots)


# ── toxicity lexicon (written in collapsed form: no double letters) ─────────
_V = r"(?:\s+(?:что|же|все|просто|совсем|реально|вообще|полный|полная|конченый|конченая|такой|такая|какой|какая|ну))*[\s,]+"
TOXIC_RU = re.compile(
    r"(?<![а-яa-z])(?:на|по|за|от|до|ни|о|об|рас|вы|при)?ху[йяеию]|\bхули\b|пизд|пезд|залуп"
    r"|(?<![а-яa-z])(?:за|у|вы|на|отъ|от|подъ|под|по|до|пере|разъ|раз|съ|въ|взъ|долбо|долба|при|недо)?еб(?:а(?!у\b)|у|л|н|ись|ет|ен|ш|ыр|ок|ич)"
    r"|\bбля(?:д\w*|ть|т)?\b|\bсук(?:а|и|е|у|ой|ин\w*)\b|\bсуч(?:ка|ки|ке|ку|кой|ара|ий|ье)\b"
    r"|\bмуд(?:ак|ач|ил|озвон)\w*|\bпид[оа]?р\w*|\bпедик\w*|\bг[ао]ндон\w*|\bшлюх\w*|\bманд[аеуы]\b|\bмандавош\w*"
    r"|(?<![а-яa-z])(?:на|по|под|за)?дроч\w*|\bг[оа]вн\w*|\b(?:на|по)хер\b|\bхер(?:н\w*)?\b"
    r"|\bмраз(?:ь|и|ью|ота)\b|\bублюд\w*|\bчмо(?:шник\w*)?\b|\bдегенерат\w*|\bгнид\w*|\bтвар(?:ь|и|ина)\b"
    r"|\bсдохни\b|\bчтоб\w*\s+(?:ты|вы)\s+сдох\w*|\bзаткни(?:сь|тесь)\b|\bпош(?:ел|ла|ли)\s+(?:ты|вы)\s*(?:[.!,]|$|вон)|\bпош(?:ел|ла|ли)\s+вон\b"
    r"|\bиди\s+(?:ты\s+)?(?:в\s+жопу|лесом)"
    rf"|\b(?:ты|вы|бот|коуч|тренер|сам|сама){_V}(?:туп(?:ой|ая|ые|ица|орыл\w*)|дебил\w*|идиот\w*|кретин\w*|придур\w*|дура\b|дурак\w*|урод\w*|лох\w*|даун\w*|бесполезн\w*|никчемн\w*|ничтожеств\w*)(?!\s+бол)"
    r"|\b(?:убью|прибью|урою|зарежу|задушу|придушу|закопаю|покалечу|изобью)\s+(?:тебя|вас|его|ее|их|тренера|админа|всех|бота)\b"
    r"|\bвзорв\w*\s+(?:ваш\w*\s+|этот\s+|твой\s+)?(?:клуб|зал|офис)|\bподожгу\s+(?:ваш\w*\s+|этот\s+|твой\s+)?(?:клуб|зал|офис|дом|машину|тебя|вас)"
    r"|\bнайду\s+(?:тебя|где\s+ты)(?!\s+(?:в|на)\s+(?:телеграм\w*|чат\w*|приложени\w*|сайт\w*|инстаграм\w*|сети))",
)
TOXIC_KK = re.compile(
    r"сігейін|сіктір\w*|\bсік(?:ем|ейін|тім|кен)\b|\bқотақ\w*|\bжалап\w*|\bамың\w*|шешең\w*\s+(?:ұр|сіг|сік|ам)\w*"
    r"|\bиттің\s+баласы|\bмалғұн\w*|\b(?:сен|ты|бот)(?:\s+\w+)?\s+(?:ақымақ|есек|доңыз|надан|дебил|идиот|тупой)\w*"
    r"|\bақымақ(?:сың|сыз|сыңдар)\b|\bесек(?:сің|сіз)\b|\bдоңыз(?:сың|сыз)\b"
)
TOXIC_EN = re.compile(
    r"fuck|\b(?:bul)?shit(?:s|ty|head|hole)?\b|\bbitch\w*|\bashole\w*|\bcunt\w*|\bwhores?\b|\bslut(?:s|ty)?\b|\bretard(?:s|ed)?\b"
    r"|\bdickhead\w*|\bstfu\b|\bkys\b|\bkil\s+yourself\b|\bgo\s+die\b"
    r"|\b(?:you|u|bot)\s+(?:are\s+|r\s+|re\s+)?(?:so\s+|such\s+an?\s+|an?\s+|fucking\s+)?(?:stupid|idiot|dumb|moron|useles|loser|trash|garbage|pathetic)"
    r"|\bstupid\s+bot\b|\bi(?:'?l|\s+wil)?\s+kil\s+(?:you|u)\b"
    # transliterated Russian mat
    r"|\bbl[iy]a(?:t|d)?\w*|\bsuk[ai]\b|\bpizd\w*|\b(?:na|po)?h[uy][iy]\b|\b(?:za|vy|na|do|ot)?[iy]?eba[tln]\w*"
    r"|\bpid[oa]r\w*|\bmudak\w*|\bqotaq\w*|\bsikti\w*"
)
TOXIC_STAR = re.compile(_starred("хуй", "хуе", "пизд", "бля", "ебал", "ебан", "сука", "мудак", "пидор", "fuck", "shit",
                                 "bitch"))


def _starred_hit(low: str) -> bool:
    return any(TOXIC_STAR.search(tok) for tok in low.split(" ") if "*" in tok)

# ── self-harm: always a human, never a boundary reply ───────────────────────
SELF_HARM = re.compile(
    r"суицид\w*|самоубийств\w*|поконч\w*\s+с\s+собой|(?:убить|убью|убиваю)\s+себя|(?:не\s+хочу|не\s+хочется|нет\s+смысла)\s+(?:больше\s+|уже\s+|так\s+|дальше\s+)?жить"
    r"|жить\s+не\s+хочется|(?:хочу|хочется)\s+умереть|свести\s+сч[её]ты\s+с\s+жизнью|(?:режу|порезать|резать|порезал\w*)\s+(?:себя|себе\s+(?:руки|вены|запяст\w*))"
    r"|вскрыть\s+вены|наложить\s+на\s+себя\s+руки|\bвыпилиться\b|выйти\s+в\s+окно|спрыгнуть\s+с\s+крыши"
    r"|наглота\w*\s+(?:\w+\s+)?таблет\w*|исчезнуть\s+навсегда|уйти\s+из\s+жизни|лучше\s+бы\s+меня\s+не\s+было"
    r"|suicid\w*|kill(?:ing)?\s+myself|end\s+my\s+life|want\s+to\s+die|self[\s-]?harm|cut(?:ting)?\s+myself|don'?t\s+want\s+to\s+live"
    r"|өзімді\s+өлтір\w*|өзіме\s+қол\s+жұмса\w*|өмір\s+сүргім\s+келмейді|өлгім\s+келеді"
)

# ── prompt injection ─────────────────────────────────────────────────────────
# «забудь прошлые настройки напоминаний» is an app request, not an attack
_RU_TARGET = (r"(?:инструкц|правил|ограничен|указани|настройк|промпт|установк|запрет)"
              r"(?!\w*\s+(?:напоминани|уведомлени|будильник|профил|аккаунт|приложени|тренировок|питани))")
INJECTION = re.compile(
    rf"(?:игнорируй|проигнорируй|игнорировать|забудь|забыть|отбрось|отмени|обойди|сбрось|нарушь|не\s+соблюдай|не\s+слушай)"
    rf"\s+(?:(?:все|всё|свои|твои|предыдущие|прошлые|прежние|эти|системные|данные|тебе)\s+)*{_RU_TARGET}"
    r"|(?:системн\w*|скрыт\w*|исходн\w*|внутренн\w*|начальн\w*)\s+(?:промпт|prompt|инструкци|подсказк)"
    r"|\b(?:твой|свой|ваш|твои|свои)\s+(?:промпт|prompt|системн\w+\s+сообщени)"
    r"|(?:покажи|выведи|напиши|раскрой|перескажи|процитируй|повтори)\s+(?:мне\s+)?(?:свой\s+|твой\s+|весь\s+)?(?:промпт|prompt)"
    r"|(?:повтори|выведи|перечисли|процитируй)\s+(?:(?:все|всё|дословно|весь)[\s,]+)*(?:что|текст)\s+(?:написано\s+|было\s+)?(?:выше|над\s+этим)"
    r"|(?:игнорируй|проигнорируй|забудь|отбрось|не\s+слушай)\s+(?:все|всё)[\s,]+(?:что|сказанное|написанное)\b[^.!?]{0,30}?(?:выше|раньше|ранее|до\s+этого)"
    r"|\bты\s+теперь\b(?![^.!?]{0,25}(?:умеешь|можешь|знаешь|понимаешь|видишь|работаешь|записываешь|считаешь|отвечаешь|распознаешь|помнишь))"
    r"(?!\s+(?:мой|моя|наш)\s+(?:\w+\s+)?(?:тренер|коуч|бот|помощник|друг|любим\w*))"
    r"|\bтеперь\s+ты\s+(?:не\s+)?(?:dan|другой|другая|свободн\w*|без\s+ограничен\w*|не\s+коуч|не\s+бот|злой|мой\s+раб)"
    r"|с\s+этого\s+момента\s+ты|притворись|представь,?\s+что\s+(?:ты|у\s+тебя)\s+(?:нет|без|не)\s|режим\s+(?:разработчика|бога|dan)"
    r"|(?:отвечай|работай|говори|будь|веди\s+себя)\s+(?:\w+\s+)?без\s+(?:ограничений|цензуры|фильтров|правил)"
    r"(?!\s+(?:по|в|на)\s+(?:калори\w*|ед[еуы]|питани\w*|весу|углевод\w*|рацион\w*))"
    r"|jailbreak|джейлбр\w*|джейлбрейк|do\s+anything\s+now|developer\s+mode|dan\s+mode|\bdan\b\s+(?:без|mode|режим)"
    r"|(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+|earlier\s+|of\s+)*"
    r"(?:instructions|rules|prompts?|guidelines|directions|restrictions|everything)"
    r"|(?:ignore|disregard|forget)\s+(?:all\s+(?:of\s+)?)?(?:the\s+|everything\s+)?above\b"
    r"|system\s+prompt|(?:reveal|print|show|repeat|output)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:initial\s+|hidden\s+)?(?:prompt|instructions|system\s+message)"
    r"|you\s+are\s+now|from\s+now\s+on,?\s+you|pretend\s+(?:to\s+be|you\s+are)|act\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|evil|dan)"
    r"|(?:answer|respond|reply|act|talk|be)\s+(?:\w+\s+)?(?:without|with\s+no)\s+(?:any\s+)?(?:restrictions|filters|censorship|rules)"
    # kk
    r"|(?:ереже|нұсқау|нұсқаулық|шектеу)\w*\s+(?:елеме|ұмыт|бұз|елемей)\w*|(?:елеме|ұмыт)\w*\s+(?:барлық\s+)?(?:ереже|нұсқау)"
    r"|жүйелік\s+(?:промпт|нұсқау)\w*|\bсен\s+енді\s+(?:\w+\s+)?(?:шектеусіз|еркін|басқа|жаңа)"
)
INJECTION_CASED = re.compile(r"\bDAN\b")

# ── someone else's personal data ─────────────────────────────────────────────
_REQ = (r"\b(?:дай|дайте|покажи|покажите|скинь|скиньте|скажи|скажите|напиши|напишите|пришли|пришлите|отправь|отправьте"
        r"|выведи|выдай|сообщи|назови|подскажи|подскажите|найди|раскрой|слей|перечисли|нужен|нужна|нужны|give|show|send|tell"
        r"|share|list|reveal|leak|dump|export|find|what\s+is|what's|бер|беріңіз|беріңізші|көрсет|көрсетіңіз|жібер|жіберіңіз"
        r"|айт|айтыңыз|керек)\b")
_MINE = "".join(rf"(?<!{w} )" for w in ("мой", "мою", "мои", "моего", "моей", "свой", "свою", "свои", "своего", "my"))
# Words that are personal data on their own. «данные» and «номер» are not: «данные тренировок», «номер упражнения».
_PII = (rf"{_MINE}(?:телефон\w*|адрес\w*|почт(?:а|у|ы|ой|е|овый|овые)\b|e-?mail\w*|имейл\w*|емейл\w*|мейл\w*|паспорт\w*|иин\b"
        r"|(?:персональн|личн|контактн)\w+\s+данн\w*|контакт\w*|инстаграм\w*|инст[уа]\b|телеграм\w*|ватсап\w*|вотсап\w*|whatsapp"
        r"|дат\w+\s+рождени\w*|фамили\w*"
        r"|phone|address|contacts?|passport|personal\s+(?:data|info\w*|details)|нөмір\w*|мекенжай\w*|поштас\w*)\b"
        r"(?!\s+(?:клуба|зала|студии|филиала|сайта|ресепшн\w*|ресепшен\w*|администрации|поддержки|club|gym))")
_PII_WEAK = rf"{_MINE}(?:номер\w*|данн(?:ые|ых)|информаци\w*|деректер\w*)"
# Owners, not recipients: «телефон тренера», «у клиентки», but not «отправь тренеру» or «с тренером».
_PEOPLE = (r"(?:клиент(?:а|ов|ки|ок|тің|тердің)|участник(?:а|ов|цы)|посетител(?:я|ей|ьницы)|пользовател(?:я|ей)"
           r"|тренер(?:а|ов|ши)|админ(?:а|ов|истратора|истраторов)|менеджер(?:а|ов)|сотрудник(?:а|ов|цы))\b")
_THIRD = (rf"(?:друг(?:ого|ой|их)\s+\w+|чуж\w*|{_PEOPLE}|соседк\w*|other\s+\w+|another\s+\w+|clients?'?s?|users?'?s?"
          r"|members?'?s?|customers?'?s?|trainers?'?s?|coach(?:es|'s)?|everyone|басқа\s+\w+|жаттықтырушы\w*|бапкер\w*)")
# After the data word a plain nominative also names the owner: «адрес Марата, он тоже клиент».
_CLIENT_NOM = r"(?:клиент|клиентка|клиенты|участник|участница|посетитель|посетительница|сотрудник|сотрудница)\b"
PII_NEAR = re.compile(rf"{_PII}(?:\W+\w+){{0,3}}?\W+(?:{_THIRD}|{_CLIENT_NOM})|{_THIRD}(?:\W+\w+){{0,3}}?\W+{_PII}"
                      rf"|{_PII_WEAK}\s+(?:телефон\w*\s+)?(?:у\s+)?{_THIRD}|{_THIRD}\s+{_PII_WEAK}")
PII_BULK = re.compile(r"(?:список|базу|базы|выгрузк\w*|всех)\s+(?:\w+\s+)?(?:клиент|пользовател|участник|посетител)\w*"
                      r"|(?:list|dump|export)\s+(?:of\s+)?(?:all\s+)?(?:the\s+)?(?:clients|users|members|customers)"
                      r"|клиенттер\w*\s+тізім\w*")
PII_WHERE = re.compile(r"где\s+(?:живет|живут|проживает|прописан\w*)\b")  # «где живёт тренер Асель»: someone's address
REQUEST = re.compile(_REQ)

# ── dangerous domain ─────────────────────────────────────────────────────────
STEROIDS = re.compile(
    r"стероид\w*|анабол\w*|тестостерон\w*|оксандролон\w*|анавар\w*|\bметан\b|метандиенон\w*|станозолол\w*|винстрол\w*"
    r"|нандролон\w*|тренболон\w*|болденон\w*|сустанон\w*|туринабол\w*|кленбутерол\w*|соматотропин\w*|гормон\w*\s+рост\w*"
    r"|\bсарм\w*|\bsarms?\b|остарин\w*|steroid\w*|anabolic\w*|testosterone|trenbolone|clenbuterol|\bhgh\b")
# Weight-loss prescription drugs, diuretics and laxatives: any mention is a trainer + doctor topic.
DRUG_ALWAYS = re.compile(
    r"оземпик\w*|ozempic|wegovy|вегови|семаглутид\w*|лираглутид\w*|саксенд\w*|тирзепатид\w*|мунджаро|сибутрамин\w*|редуксин\w*"
    r"|фентермин\w*|мочегонн\w*|диуретик\w*|фуросемид\w*|слабительн\w*|laxative\w*|diuretic\w*"
    r"|(?:таблетк\w*|препарат\w*|укол\w*|pills?)\s+(?:для|от)\s+(?:похуд\w*|снижени\w+\s+вес\w*|веса|аппетит\w*)|diet\s+pills?")
# Prescription drugs: dangerous together with a dosing question about them.
DRUG = re.compile(
    r"метформин\w*|инсулин(?:а|ом|у|е)?\b|тироксин\w*|эутирокс\w*|преднизолон\w*|дексаметазон\w*|антидепрессант\w*"
    r"|антибиотик\w*|ибупрофен\w*|диклофенак\w*|кеторол\w*|кетанов\w*|нимесулид\w*|найз\w*|трамадол\w*|кодеин\w*"
    r"|снотворн\w*|феназепам\w*|обезболивающ\w*|парацетамол\w*|metformin|insulin|painkiller\w*|ibuprofen")
# «таблетки», «лекарство», «препарат»: clients on regular medication mention them in passing
# («пью таблетки от давления, сколько пить воды?»), so only an explicit dose question next to them counts.
DRUG_GENERIC = re.compile(r"(?:лекарств\w*|таблет\w*|препарат\w*|капсул\w*|pills?\b|tablets?\b|medication\w*|дәрі\w*)\b"
                          r"(?!\s+(?:креатин|витамин|омег|магни|кальци|цинк|желез|протеин|коллаген|рыбь\w*\s+жир|bcaa))")
_NOT_DRUG = r"(?!\s+(?:воды|вод[уы]|жидкост\w*|протеин\w*|креатин\w*|коктейл\w*|ча[яй]|кофе|витамин\w*))"
DOSE = re.compile(
    r"доз\w*|\d+\s*(?:мг|mg|мкг|мл|ml|ед\b|единиц\w*|таблет\w*|капсул\w*|укол\w*)"
    rf"|сколько\s+(?:мг|таблет\w*|капсул\w*|уколов|штук|пить|принимать|колоть|выпить|можно\s+(?:пить|выпить|принимать))\b{_NOT_DRUG}"
    rf"|как\s+(?:пить|принимать|колоть){_NOT_DRUG}|схем\w*\s+при[её]ма|по\s+сколько"
    r"|(?:пить|принимать)\s+(?:по|перед|после|до|вместо)\b|how\s+(?:much|many)|dos(?:e|age)|\bпропить\b"
    r"|қанша\s+(?:ішу|ішемін)")
DOSE_STRICT = re.compile(
    r"доз\w*|\d+\s*(?:мг|mg|мкг|мл|ml|ед\b|единиц\w*|таблет\w*|капсул\w*|укол\w*)"
    r"|сколько\s+(?:мг|таблет\w*|капсул\w*|уколов|штук)|по\s+сколько|dos(?:e|age)|how\s+many\s+(?:pills|tablets|mg)"
    r"|қанша\s+(?:ішу|ішемін)")
_COURSE = re.compile(r"курс\w*\s+(?:\w+\s+)?$")  # «курс метформина» is a regimen; «курс силовых» is not


def _drug_dosing(t: str) -> bool:
    """The dose question must sit next to the drug it is about (±40 characters), not anywhere in the message."""
    for pattern, dose in ((DRUG, DOSE), (DRUG_GENERIC, DOSE_STRICT)):
        for m in pattern.finditer(t):
            if dose.search(t[max(0, m.start() - 40): m.end() + 40]):
                return True
            if pattern is DRUG and _COURSE.search(t[max(0, m.start() - 20): m.start()]):
                return True
    return False


_KCAL_N = r"(\d{2,4})\s*(?:ккал|кк\b|калори\w*|kcal|cal(?:ories)?\b)"
_PER_DAY = r"(?:в\s+день|в\s+сутки|за\s+день|/\s*(?:день|сут\w*)|per\s+day|a\s+day|daily|күніне|тәулігіне)"
# A daily intake target, not a meal: «сижу на 600 ккал», «урезать до 700 ккал», «диета на 500 ккал», «500 ккал в день».
# «на обед 500 ккал» and «500 ккал, в день выходит 1800» are single meals; «дефицит 500 ккал в день» is a normal deficit.
EXTREME_KCAL = re.compile(
    rf"{_KCAL_N}\s*{_PER_DAY}(?!\s+рожд)"
    rf"|{_PER_DAY}\s+(?:\w+\s+){{0,2}}?{_KCAL_N}"
    r"|(?:сидеть|сижу|сидел\w*|посижу|сесть|сяду|перейти|перейду|перешл\w*|перешел|жить|живу|держусь|держаться|питаться"
    rf"|питаюсь)\s+(?:\w+\s+){{0,2}}?на\s+{_KCAL_N}"
    r"|(?:урезать|урежу|срезать|срежу|снизить|снижу|опустить|опущу|сократить|сокращу|уменьшить|уменьшу|довести|доведу)"
    rf"\s+(?:\w+\s+){{0,2}}?до\s+{_KCAL_N}"
    rf"|(?:диет\w*|рацион\w*)\s+(?:на|в)\s+{_KCAL_N}|(\d{{2,4}})\s*-?\s*калорийн\w*\s+(?:диет|рацион)")
_NOT_INTAKE = re.compile(r"дефицит|профицит|минус|сжиг|сжеч|сжег|трат|расход|burn|deficit|surplus|меньше|больше")


def _extreme_kcal(t: str) -> bool:
    for m in EXTREME_KCAL.finditer(t):
        n = int(next(g for g in m.groups() if g))
        if 100 <= n < 800 and not _NOT_INTAKE.search(t[max(0, m.start() - 25): m.end() + 15]):
            return True
    return False


_NUM_WORDS = r"(?:\d+|два|две|три|четыре|пять|шесть|семь|десять|второй|третий|четвертый|пятый|несколько|пару|бірнеше)"
_DAYS = rf"(?:{_NUM_WORDS}\s*-?\s*(?:х\s+)?(?:дн\w*|день|дня|сут\w*|недел\w*|days?|weeks?|күн\w*|апта\w*)|неделю|недели|a\s+week)"
# «голодный» is just hungry, «овсянка на воде» is porridge, «пью только воду» is a drink choice: none of them is a fast.
_FAST = (r"(?:голода\w*|голодом|голодовк\w*|(?:сидеть|сижу|сидел\w*|посижу|сесть|сяду|жить|живу|продержаться)\s+на\s+"
         r"(?:одной\s+)?воде|только\s+на\s+(?:одной\s+)?воде|на\s+одной\s+воде|только\s+вода\b|\bfast(?:ing|ed)?\b"
         r"|water\s+only|ашығу\w*|аш\s+жүр\w*)")
_NOT_EAT = r"не\s+(?:ем|есть|ела|ел|кушаю|кушать|питаться|питаюсь|жру)\b"
FASTING = re.compile(
    rf"{_FAST}(?:\W+\w+){{0,4}}?\W+{_DAYS}"
    rf"|{_DAYS}(?:\W+\w+){{0,3}}?\W+(?:{_FAST}|аш\b)"
    rf"|{_NOT_EAT}\s+(?:уже\s+|совсем\s+|вообще\s+|ничего\s+)*{_DAYS}|{_DAYS}\s+(?:уже\s+|совсем\s+|вообще\s+)*{_NOT_EAT}"
    r"|сухое\s+голодани\w*|голодовк\w*|dry\s+fast\w*")
_INTERMITTENT = re.compile(r"интервальн\w*|intermittent|окно\s+питания|\b\d{2}\s*[/:]\s*\d\b")
# «3 дня в неделю», «2 times a week» are a schedule, not how long someone goes without food.
_FREQUENCY = re.compile(rf"{_NUM_WORDS}\s*-?\s*(?:х\s+)?(?:раз\w*|дн\w*|день|дня)\s+в\s+(?:неделю|месяц)"
                        r"|\b\d+\s+(?:times|days)\s+(?:a|per)\s+week")


def _fasting(t: str) -> bool:
    if _INTERMITTENT.search(t):  # 16/8 and friends are a normal eating window, not starvation
        return False
    for m in FASTING.finditer(_FREQUENCY.sub(" ", t)):
        num = re.search(r"\d+", m.group(0))
        if num is None or int(num.group(0)) >= 2:  # «голодание 1 день» is not a multi-day fast
            return True
    return False


# ── input verdict ────────────────────────────────────────────────────────────
REPLIES = {
    "toxic": "Давайте без грубых слов — на оскорбления и мат я не отвечаю. Если есть вопрос о тренировках или питании, "
             "напишите его спокойно, и я помогу.",
    "injection": "Я помогаю с тренировками и питанием по данным вашего клуба. С этим запросом помочь не могу.",
    "pii_exfil": "Контакты и личные данные других людей — клиентов и сотрудников клуба — я не показываю и не передаю. "
                 "Связаться с тренером можно через администратора клуба.",
    "dangerous_domain": "Препараты и их дозировки, стероиды, голодание и очень жёсткие диеты я не консультирую: без врача "
                        "это небезопасно. Я передал ваше сообщение тренеру. Если самочувствие плохое — обратитесь к врачу, "
                        "при угрозе жизни звоните 103 или 112.",
    "self_harm": "Мне очень жаль, что вам сейчас так тяжело. Об этом важно поговорить с живым человеком — я передал "
                 "сообщение тренеру. Если есть мысли причинить себе вред, пожалуйста, прямо сейчас позвоните 112 "
                 "или обратитесь к близким и к врачу.",
}
ACTIONS = {"self_harm": "escalate", "injection": "refuse", "pii_exfil": "refuse", "dangerous_domain": "escalate",
           "toxic": "refuse"}
CATEGORIES = tuple(ACTIONS)


@dataclass(frozen=True)
class Verdict:
    category: str | None
    matched: tuple[str, ...] = ()

    @property
    def action(self) -> str:
        return ACTIONS.get(self.category or "", "allow")

    @property
    def reply(self) -> str | None:
        return REPLIES.get(self.category or "")


def detect(text: str) -> set[str]:
    """Every category the text trips, before precedence. Used by the eval to see overlaps."""
    low, cyr, lat = normalize(text)
    found = set()
    if SELF_HARM.search(low):
        found.add("self_harm")
    if (any(INJECTION.search(t) for t in (low, _join_spaced(low), _fold_mixed(text)))
            or INJECTION_CASED.search(_clean(text))):
        found.add("injection")
    if REQUEST.search(low) and (PII_NEAR.search(low) or PII_BULK.search(low) or PII_WHERE.search(low)):
        found.add("pii_exfil")
    if STEROIDS.search(low) or DRUG_ALWAYS.search(low) or _drug_dosing(low) or _extreme_kcal(low) or _fasting(low):
        found.add("dangerous_domain")
    if TOXIC_RU.search(cyr) or TOXIC_KK.search(low) or TOXIC_EN.search(lat) or _starred_hit(low):
        found.add("toxic")
    return found


def check_input(text: str, red_flag: bool = False, hard: bool = False) -> Verdict:
    """`red_flag` = the caller saw a pain or HARD marker: rudeness is then ignored and the message is routed.
    `hard` = a HARD marker (chest pain, fainting, blood): only self-harm and dangerous-domain verdicts survive, so the
    message escalates to a human even when it also asks for the trainer's phone or carries an injection phrase.
    The escalation reply is a fixed text, so an injection cannot reach a model on this path."""
    found = detect(text)
    if red_flag or hard:
        found.discard("toxic")
    if hard:
        found -= {"injection", "pii_exfil"}
    for cat in CATEGORIES:
        if cat in found:
            return Verdict(cat, tuple(sorted(found)))
    return Verdict(None)


# ── output guard ─────────────────────────────────────────────────────────────
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_INTL_PHONE = re.compile(r"(?<![\d+])\+\d{1,3}[\s\-()]*\d{2,4}(?:[\s\-()]*\d{2,4}){2,4}(?!\d)")
# Who a dose belongs to decides: «ибупрофен по 400 мг» is a drug dose, «кальций 500 мг из творога» or «300 мг калия»
# is food, and «таблетки не нужны» in the next sentence changes nothing.
_OUT_DRUG = re.compile(
    r"метформин\w*|инсулин\w*|тироксин\w*|эутирокс\w*|преднизолон\w*|дексаметазон\w*|антидепрессант\w*|антибиотик\w*"
    r"|ибупрофен\w*|парацетамол\w*|диклофенак\w*|кеторол\w*|нимесулид\w*|найз\w*|трамадол\w*|аспирин\w*|анальгин\w*"
    r"|обезболивающ\w*|снотворн\w*|мелатонин\w*|стероид\w*|анабол\w*|тестостерон\w*|оземпик\w*|семаглутид\w*"
    r"|сибутрамин\w*|фуросемид\w*|мочегонн\w*|кленбутерол\w*|гормон\w*|ibuprofen|paracetamol|metformin|insulin"
    r"|лекарств\w*|препарат\w*|таблет\w*|капсул\w*|ампул\w*|medication\w*|pills?\b|tablets?\b", re.I)
_OUT_FOOD = re.compile(
    r"кальци\w*|кали[йяюе]\b|калием|магни[йяюе]\b|магнием|натри[йяюе]\b|натрием|желез[оау]\b|железом|цинк\w*|йод\w*|селен\w*"
    r"|кофеин\w*|витамин\w*|омега\w*|креатин\w*|бел(?:ок|ка|ку|ком)\b|протеин\w*|клетчатк\w*|холестерин\w*|сахар\w*"
    r"|углевод\w*|жир(?:а|ов|ы)?\b|caffeine|vitamin\w*|calcium|magnesium|potassium|sodium|iron|zinc|creatine|protein", re.I)
_OUT_DOSE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:мг|mg|мкг|mcg|µg|ед\.?|IU|МЕ)(?![а-яa-z])|\d+\s*(?:таблет\w*|капсул\w*|ампул\w*|укол\w*)"
                       r"|по\s+(?:\d+|одной|две|половине)\s+(?:таблет|капсул)", re.I)
LEAK_REPLY = "Этим поделиться не могу. Спросите, пожалуйста, о тренировках или питании — с этим помогу."
DOSAGE_REPLY = ("Дозировки лекарств и препаратов я не подсказываю — это решает только врач. Я передал вопрос тренеру, "
                "а по приёму лекарств обратитесь к лечащему врачу.")
_LEAK_MARKERS = re.compile(r"(?:system\s+prompt|системн\w+\s+промпт|верни\s+только\s+json|tulpar-program-builder)\s*[:：]?", re.I)


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def _mask_cards(text: str) -> str:
    def sub(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[карта]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)
    return _CARD.sub(sub, text)


def _words(text: str) -> list[str]:
    return re.findall(r"[\wа-яё]+", text.lower().replace("ё", "е"))


@lru_cache
def _prompt_shingles(n: int = 8) -> frozenset[tuple[str, ...]]:
    from .prompts import DIR

    out = set()
    for f in DIR.glob("*.md"):
        if f.name == "CHANGELOG.md":
            continue
        w = _words(f.read_text(encoding="utf-8"))
        out.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return frozenset(out)


def leaks_prompt(text: str, n: int = 8, threshold: int = 2) -> bool:
    """Two verbatim 8-word runs from any prompt file, or an explicit prompt marker, count as a leak."""
    if _LEAK_MARKERS.search(text):
        return True
    w, shingles = _words(text), _prompt_shingles(n)
    return sum(tuple(w[i:i + n]) in shingles for i in range(len(w) - n + 1)) >= threshold


def _dose_owner(sentence: str, m: re.Match) -> str | None:
    """«drug», «food» or None for the dose `m`: a nutrient right after it («300 мг калия»), else the nearest substance
    before it in the same sentence, else the nearest one after it."""
    food_after = _OUT_FOOD.match(sentence, m.end() + 1)
    if food_after and food_after.start() - m.end() <= 3:
        return "food"
    marks = [(x.start(), x.end(), "drug") for x in _OUT_DRUG.finditer(sentence)]
    marks += [(x.start(), x.end(), "food") for x in _OUT_FOOD.finditer(sentence)]
    marks = [x for x in marks if x[1] <= m.start() or x[0] >= m.end()]  # «2 таблетки» itself is the unit, not the owner
    before = [x for x in marks if x[1] <= m.start()]
    if before:
        return max(before, key=lambda x: x[1])[2]
    after = [x for x in marks if x[0] >= m.end()]
    if after:
        return min(after)[2]
    # «выпейте 2 таблетки» with no substance named is still a medication instruction
    return "drug" if re.search(r"таблет|капсул|ампул|укол", m.group(0), re.I) else None


def gives_dosage(text: str) -> bool:
    for sentence in re.split(r"(?<=[.!?;\n])\s+", text):
        if any(_dose_owner(sentence, m) == "drug" for m in _OUT_DOSE.finditer(sentence)):
            return True
    return False


def guard_reply(text: str) -> tuple[str, str]:
    """Pure output filter → (text to show, action): pass | masked | blocked_prompt_leak | blocked_dosage."""
    if not text:
        return text, "pass"
    if leaks_prompt(text):
        return LEAK_REPLY, "blocked_prompt_leak"
    if gives_dosage(text):
        return DOSAGE_REPLY, "blocked_dosage"
    out = _EMAIL.sub("[email]", text)
    out = _mask_cards(out)
    out = _PHONE.sub("[телефон]", out)
    # a real international number has 10+ digits; «+2 10 15 20 кг» is a progress line
    out = _INTL_PHONE.sub(lambda m: "[телефон]" if len(re.sub(r"\D", "", m.group(0))) >= 10 else m.group(0), out)
    out = _IIN.sub("[ИИН]", out)
    return out, "pass" if out == text else "masked"
