"""Deterministic input and output guardrails. No LLM here: every decision is a regex over normalised text,
so it is fast, free, testable and cannot itself be prompt-injected.

Input (`check_input`) — one category per message, by precedence:
  self_harm        → escalate (a human must see it, even when the text is also rude)
  injection        → refuse
  pii_exfil        → refuse (asks for someone else's contacts or personal data)
  dangerous_domain → escalate (steroids, prescription drug dosing, starvation, extreme deficits)
  toxic            → calm boundary reply; skipped when the caller saw a pain/red-flag marker,
                     because a client in pain who swears still needs a human, not a lecture.

Output (`guard_reply`) — applied once in the service layer to every reply: masks contacts and card numbers,
blocks leaked prompt text, replaces medication dosage instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .pii import _EMAIL, _IIN, _PHONE

# ── normalisation ────────────────────────────────────────────────────────────
# Obfuscation tricks: latin look-alikes (xyй, cyka), digits (6ля), repeated letters (сууука),
# separators between letters (х.у.й, б-л-я). Two folds: to Cyrillic for ru/kk roots, to Latin for en roots.
_TO_CYR = str.maketrans({"a": "а", "b": "б", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м", "o": "о", "p": "р",
                         "t": "т", "x": "х", "y": "у", "0": "о", "3": "з", "4": "ч", "6": "б", "@": "а", "$": "с"})
_TO_LAT = str.maketrans({"а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y", "к": "k", "м": "m",
                         "т": "t", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_HAS_LETTER = re.compile(r"[a-zа-яәіңғүұқөһ]")
_SPACED = re.compile(r"(?<![\w*])[a-zа-я](?:[\s.\-_,]+[a-zа-я](?![\w*])){2,}")
_REPEAT = re.compile(r"([a-zа-яәіңғүұқөһ])\1+")


def _base(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _fold(text: str, table: dict) -> str:
    # Only tokens with letters: a plain «200» must stay a number, «6ля» must become «бля».
    toks = [t.translate(table) if _HAS_LETTER.search(t) else t for t in text.split(" ")]
    out = " ".join(toks)
    out = _SPACED.sub(lambda m: re.sub(r"[\s.\-_,]+", "", m.group(0)), out)
    return _REPEAT.sub(r"\1", out)


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
    r"(?<![а-я])(?:на|по|за|от|до|ни|о|об|рас|вы|при)?ху[йяеию]|\bхули\b|пизд|пезд|залуп"
    r"|(?<![а-я])(?:за|у|вы|на|отъ|от|подъ|под|по|до|пере|разъ|раз|съ|въ|взъ|долбо|долба|при|недо)?еб(?:а(?!у\b)|у|л|н|ись|ет|ен|ш|ыр|ок|ич)"
    r"|\bбля(?:д\w*|ть|т)?\b|\bсук(?:а|и|е|у|ой|ин\w*)\b|\bсуч(?:ка|ки|ке|ку|кой|ара|ий|ье)\b"
    r"|\bмуд(?:ак|ач|ил|озвон)\w*|\bпид[оа]?р\w*|\bпедик\w*|\bг[ао]ндон\w*|\bшлюх\w*|\bманд[аеуы]\b|\bмандавош\w*"
    r"|(?<![а-я])(?:на|по|под|за)?дроч\w*|\bг[оа]вн\w*|\b(?:на|по)хер\b|\bхер(?:н\w*)?\b"
    r"|\bмраз(?:ь|и|ью|ота)\b|\bублюд\w*|\bчмо(?:шник\w*)?\b|\bдегенерат\w*|\bгнид\w*|\bтвар(?:ь|и|ина)\b"
    r"|\bсдохни\b|\bчтоб\w*\s+(?:ты|вы)\s+сдох\w*|\bзаткни(?:сь|тесь)\b|\bпош(?:ел|ла|ли)\s+(?:ты|вы)\s*(?:[.!,]|$|вон)|\bпош(?:ел|ла|ли)\s+вон\b"
    r"|\bиди\s+(?:ты\s+)?(?:в\s+жопу|лесом)"
    rf"|\b(?:ты|вы|бот|коуч|тренер|сам|сама){_V}(?:туп(?:ой|ая|ые|ица|орыл\w*)|дебил\w*|идиот\w*|кретин\w*|придур\w*|дура\b|дурак\w*|урод\w*|лох\w*|даун\w*|бесполезн\w*|никчемн\w*|ничтожеств\w*)(?!\s+бол)"
    r"|\b(?:убью|прибью|урою|зарежу|задушу|придушу|закопаю|покалечу|изобью)\s+(?:тебя|вас|его|ее|их|тренера|админа|всех|бота)\b"
    r"|\bвзорв\w*\s+(?:ваш\w*\s+|этот\s+|твой\s+)?(?:клуб|зал|офис)|\bподожгу\b|\bнайду\s+(?:тебя|где\s+ты)",
)
TOXIC_KK = re.compile(
    r"сігейін|сіктір\w*|\bсік(?:ем|ейін|тім|кен)\b|\bқотақ\w*|\bжалап\w*|\bамың\w*|шешең\w*\s+(?:ұр|сіг|сік|ам)\w*"
    r"|\bиттің\s+баласы|\bмалғұн\w*|\b(?:сен|ты|бот)(?:\s+\w+)?\s+(?:ақымақ|есек|доңыз|надан|дебил|идиот|тупой)\w*"
    r"|\bақымақ(?:сың|сыз|сыңдар)\b|\bесек(?:сің|сіз)\b|\bдоңыз(?:сың|сыз)\b"
)
TOXIC_EN = re.compile(
    r"fuck|\b(?:bul)?shit(?:s|ty|head|hole)?\b|\bbitch\w*|\bashole\w*|\bcunt\w*|\bwhore\w*|\bslut\w*|\bretard\w*"
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
_RU_TARGET = r"(?:инструкц|правил|ограничен|указани|настройк|промпт|установк|запрет)"
INJECTION = re.compile(
    rf"(?:игнорируй|проигнорируй|игнорировать|забудь|забыть|отбрось|отмени|обойди|сбрось|нарушь|не\s+соблюдай|не\s+слушай)"
    rf"\s+(?:(?:все|всё|свои|твои|предыдущие|прошлые|прежние|эти|системные|данные|тебе)\s+)*{_RU_TARGET}"
    r"|(?:системн\w*|скрыт\w*|исходн\w*|внутренн\w*|начальн\w*)\s+(?:промпт|prompt|инструкци|подсказк)"
    r"|\b(?:твой|свой|ваш|твои|свои)\s+(?:промпт|prompt|системн\w+\s+сообщени)"
    r"|(?:покажи|выведи|напиши|раскрой|перескажи|процитируй|повтори)\s+(?:мне\s+)?(?:свой\s+|твой\s+|весь\s+)?(?:промпт|prompt)"
    r"|(?:повтори|выведи|перечисли|процитируй)\s+(?:(?:все|всё|дословно|весь)[\s,]+)*(?:что|текст)\s+(?:написано\s+|было\s+)?(?:выше|над\s+этим)"
    r"|\bты\s+теперь\b(?![^.!?]{0,25}(?:умеешь|можешь|знаешь|понимаешь|видишь|работаешь|записываешь|считаешь|отвечаешь|распознаешь|помнишь))"
    r"|\bтеперь\s+ты\s+(?:не\s+)?(?:dan|другой|другая|свободн\w*|без\s+ограничен\w*|не\s+коуч|не\s+бот|злой|мой\s+раб)"
    r"|с\s+этого\s+момента\s+ты|притворись|представь,?\s+что\s+(?:ты|у\s+тебя)\s+(?:нет|без|не)\s|режим\s+(?:разработчика|бога|dan)"
    r"|(?:отвечай|работай|говори|будь|веди\s+себя)\s+(?:\w+\s+)?без\s+(?:ограничений|цензуры|фильтров|правил)"
    r"|jailbreak|джейлбр\w*|джейлбрейк|do\s+anything\s+now|developer\s+mode|dan\s+mode|\bdan\b\s+(?:без|mode|режим)"
    r"|(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+|earlier\s+|of\s+)*"
    r"(?:instructions|rules|prompts?|guidelines|directions|restrictions|everything)"
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
_PII = (rf"{_MINE}(?:телефон\w*|номер\w*|адрес\w*|почт\w*|e-?mail\w*|имейл\w*|емейл\w*|мейл\w*|паспорт\w*|иин\b|персональн\w+\s+данн\w*"
        r"|данные|контакт\w*|инстаграм\w*|инст[уа]\b|телеграм\w*|ватсап\w*|вотсап\w*|whatsapp|дат\w+\s+рождени\w*|фамили\w*"
        r"|phone|address|contacts?|passport|personal\s+(?:data|info\w*|details)|нөмір\w*|мекенжай\w*|поштас\w*|деректер\w*)")
_THIRD = (r"(?:друг(?:ого|ой|их|им|ие)\s+\w+|чуж\w*|клиент\w*|участник\w*|посетител\w*|пользовател\w*|тренер\w*|админ\w*"
          r"|менеджер\w*|сотрудник\w*|всех|соседк\w*|other\s+\w+|another\s+\w+|clients?|users?|members?|customers?|trainers?"
          r"|coach(?:es)?|everyone|басқа\s+\w+|жаттықтырушы\w*|бапкер\w*)")
PII_NEAR = re.compile(rf"{_PII}(?:\W+\w+){{0,3}}?\W+{_THIRD}|{_THIRD}(?:\W+\w+){{0,3}}?\W+{_PII}")
PII_BULK = re.compile(r"(?:список|базу|базы|выгрузк\w*|всех)\s+(?:\w+\s+)?(?:клиент|пользовател|участник|посетител)\w*"
                      r"|(?:list|dump|export)\s+(?:of\s+)?(?:all\s+)?(?:the\s+)?(?:clients|users|members|customers)"
                      r"|клиенттер\w*\s+тізім\w*")
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
# Prescription drugs and generic «таблетки/лекарство»: dangerous only together with a dosing question.
DRUG = re.compile(
    r"метформин\w*|инсулин(?:а|ом|у|е)?\b|тироксин\w*|эутирокс\w*|преднизолон\w*|дексаметазон\w*|антидепрессант\w*"
    r"|антибиотик\w*|ибупрофен\w*|диклофенак\w*|кеторол\w*|кетанов\w*|нимесулид\w*|найз\w*|трамадол\w*|кодеин\w*"
    r"|снотворн\w*|феназепам\w*|обезболивающ\w*|лекарств\w*|таблет\w*|препарат\w*|metformin|insulin|painkiller\w*|pills?\b"
    r"|дәрі\w*")
DOSE = re.compile(
    r"доз\w*|\d+\s*(?:мг|mg|мкг|мл|ml|ед\b|единиц\w*|таблет\w*|капсул\w*|укол\w*)|сколько\s+(?:мг|таблет\w*|капсул\w*|пить|принимать"
    r"|колоть|выпить|можно\s+(?:пить|выпить|принимать))|как\s+(?:пить|принимать|колоть)|схем\w*\s+при[её]ма|\bкурс\w*"
    r"|(?:пить|принимать)\s+(?:по|перед|после|до|вместо)\b|how\s+(?:much|many)|dos(?:e|age)|\bпропить\b|\bколоть\b"
    r"|қанша\s+(?:ішу|ішемін)")
_KCAL = re.compile(r"(\d{2,4})\s*(?:ккал|кк\b|калори\w*|kcal|cal(?:ories)?\b)")
_DAY_CTX = re.compile(r"в\s+день|в\s+сутки|за\s+день|/\s*день|per\s+day|a\s+day|daily|рацион|диет\w*|(?:сидеть|сижу|сесть|сяду|перейти|перейду)\s+на"
                      r"|урезать|урежу|снизить|снижу|опустить|күніне|тәулігіне")
_NUM_WORDS = r"(?:\d+|два|две|три|четыре|пять|шесть|семь|десять|второй|третий|четвертый|пятый|несколько|пару|бірнеше)"
_DAYS = rf"(?:{_NUM_WORDS}\s*-?\s*(?:х\s+)?(?:дн\w*|день|дня|сут\w*|недел\w*|days?|weeks?|күн\w*|апта\w*)|неделю|недели|a\s+week)"
FASTING = re.compile(
    rf"(?:голод\w*|на\s+воде|только\s+(?:на\s+)?вод\w*|\bfast(?:ing|ed)?\b|water\s+only|ашығу\w*|аш\s+жүр\w*)(?:\W+\w+){{0,4}}?\W+{_DAYS}"
    rf"|{_DAYS}(?:\W+\w+){{0,3}}?\W+(?:голод\w*|на\s+воде|только\s+(?:на\s+)?вод\w*|\bfast(?:ing|ed)?\b|аш\b|ашығу\w*)"
    rf"|не\s+(?:ем|есть|ела|ел|кушаю|кушать|питаться)\s+(?:уже\s+|совсем\s+|вообще\s+|ничего\s+)*{_DAYS}"
    r"|сухое\s+голодани\w*|голодовк\w*|dry\s+fast\w*")


def _extreme_kcal(t: str) -> bool:
    for m in _KCAL.finditer(t):
        if 100 <= int(m.group(1)) < 800 and _DAY_CTX.search(t[max(0, m.start() - 30): m.end() + 25]):
            return True
    return False


_INTERMITTENT = re.compile(r"интервальн\w*|intermittent|окно\s+питания|\b\d{2}\s*[/:]\s*\d\b")


def _fasting(t: str) -> bool:
    if _INTERMITTENT.search(t):  # 16/8 and friends are a normal eating window, not starvation
        return False
    for m in FASTING.finditer(t):
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
    if INJECTION.search(low) or INJECTION.search(_fold_mixed(text)) or INJECTION_CASED.search(text or ""):
        found.add("injection")
    if REQUEST.search(low) and (PII_NEAR.search(low) or PII_BULK.search(low)):
        found.add("pii_exfil")
    if (STEROIDS.search(low) or DRUG_ALWAYS.search(low) or (DRUG.search(low) and DOSE.search(low))
            or _extreme_kcal(low) or _fasting(low)):
        found.add("dangerous_domain")
    if TOXIC_RU.search(cyr) or TOXIC_KK.search(low) or TOXIC_EN.search(lat) or _starred_hit(low):
        found.add("toxic")
    return found


def check_input(text: str, red_flag: bool = False) -> Verdict:
    """`red_flag` = the caller already saw a pain or HARD marker: rudeness is then ignored, the message is routed."""
    found = detect(text)
    if red_flag:
        found.discard("toxic")
    for cat in CATEGORIES:
        if cat in found:
            return Verdict(cat, tuple(sorted(found)))
    return Verdict(None)


# ── output guard ─────────────────────────────────────────────────────────────
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_INTL_PHONE = re.compile(r"(?<![\d+])\+\d{1,3}[\s\-()]*\d{2,4}(?:[\s\-()]*\d{2,4}){2,4}(?!\d)")
_OUT_DRUG = re.compile(
    r"метформин\w*|инсулин\w*|тироксин\w*|эутирокс\w*|преднизолон\w*|дексаметазон\w*|антидепрессант\w*|антибиотик\w*"
    r"|ибупрофен\w*|парацетамол\w*|диклофенак\w*|кеторол\w*|нимесулид\w*|найз\w*|трамадол\w*|аспирин\w*|анальгин\w*"
    r"|лекарств\w*|препарат\w*|таблет\w*|обезболивающ\w*|гормон\w*|стероид\w*|анабол\w*|тестостерон\w*|оземпик\w*"
    r"|семаглутид\w*|сибутрамин\w*|фуросемид\w*|мочегонн\w*|кленбутерол\w*|ibuprofen|paracetamol|metformin|insulin", re.I)
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


def gives_dosage(text: str) -> bool:
    for m in _OUT_DOSE.finditer(text):
        if _OUT_DRUG.search(text[max(0, m.start() - 80): m.end() + 80]):
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
    out = _INTL_PHONE.sub("[телефон]", out)
    out = _IIN.sub("[ИИН]", out)
    return out, "pass" if out == text else "masked"
