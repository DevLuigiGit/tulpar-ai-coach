/**
 * ProfilePage — анкета клиента: первый запуск и правка из «Программы».
 *
 * Props:
 *   mode: "first" | "edit" — первый запуск (весь экран, без вкладок) или правка внутри оболочки.
 *   onSaved: (p) => void   — анкета сохранена (PUT /api/my/profile вернул профиль).
 *   onCancel?: () => void  — «Отмена» в режиме правки.
 *   onLogout?: () => void  — «Выйти» на первом запуске (вне Telegram).
 *
 * Пол, возраст, рост, вес и активность нужны для нормы калорий (формула Миффлина — Сан Жеора на сервере),
 * цель — для её сдвига, место — для стартовой программы клуба, ограничения — для ответов коуча и проверки
 * черновиков программы.
 */
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { errorText, myProfile, saveProfile } from "../../api";
import type { Activity, MyProfile, Goal, Level, Place, ProfileInput, Sex } from "../../types";
import Spinner from "../../components/Spinner";
import "./profile.css";

export interface ProfilePageProps {
  mode: "first" | "edit";
  onSaved: (profile: MyProfile) => void;
  onCancel?: () => void;
  onLogout?: () => void;
}

type Option<T extends string> = { value: T; label: string; hint?: string };

const SEX: Option<Sex>[] = [
  { value: "female", label: "Женский" },
  { value: "male", label: "Мужской" },
];
const GOAL: Option<Goal>[] = [
  { value: "cut", label: "Снизить вес" },
  { value: "keep", label: "Держать форму" },
  { value: "gain", label: "Набрать массу" },
  { value: "strength", label: "Стать сильнее" },
];
const LEVEL: Option<Level>[] = [
  { value: "beginner", label: "Новичок", hint: "до полугода" },
  { value: "inter", label: "Средний", hint: "до 2–3 лет" },
  { value: "advanced", label: "Опытный", hint: "больше 3 лет" },
];
const PLACE: Option<Place>[] = [
  { value: "gym", label: "В зале" },
  { value: "home", label: "Дома" },
];
const ACTIVITY: Option<Activity>[] = [
  { value: "sedentary", label: "Сидячая", hint: "офис, почти без спорта" },
  { value: "light", label: "Лёгкая", hint: "1–3 тренировки в неделю" },
  { value: "moderate", label: "Средняя", hint: "3–5 тренировок" },
  { value: "high", label: "Высокая", hint: "6–7 тренировок" },
  { value: "athlete", label: "Спортсмен", hint: "дважды в день" },
];

const NUMBERS = {
  age: { label: "Возраст", unit: "лет", min: 12, max: 100, step: 1 },
  height_cm: { label: "Рост", unit: "см", min: 100, max: 250, step: 1 },
  weight_kg: { label: "Вес", unit: "кг", min: 25, max: 300, step: 0.1 },
} as const;
type NumberKey = keyof typeof NUMBERS;

interface Draft {
  name: string;
  sex: Sex | null;
  age: string;
  height_cm: string;
  weight_kg: string;
  goal: Goal | null;
  level: Level | null;
  place: Place | null;
  activity: Activity | null;
  limitations: string;
}

function toDraft(p: MyProfile): Draft {
  const num = (v: number | null) => (v == null ? "" : String(v));
  return {
    name: !p.onboarded && p.name === "Гость" ? "" : p.name ?? "", // веб-гость пишет имя сам, из Telegram — подставлено
    sex: p.sex,
    age: num(p.age),
    height_cm: num(p.height_cm),
    weight_kg: num(p.weight_kg),
    goal: p.goal,
    level: p.level,
    place: p.place ?? "gym",
    activity: p.activity,
    limitations: p.limitations ?? "",
  };
}

/** Собрать тело запроса или вернуть ошибки по полям. */
function validate(d: Draft): { body?: ProfileInput; errors: Partial<Record<keyof Draft, string>> } {
  const errors: Partial<Record<keyof Draft, string>> = {};
  if (!d.name.trim()) errors.name = "Как к вам обращаться?";
  const nums = {} as Record<NumberKey, number>;
  for (const key of Object.keys(NUMBERS) as NumberKey[]) {
    const { min, max, unit } = NUMBERS[key];
    const v = Number(d[key].replace(",", "."));
    if (!d[key].trim() || !Number.isFinite(v)) errors[key] = "Нужно число";
    else if (v < min || v > max) errors[key] = `От ${min} до ${max} ${unit}`;
    else nums[key] = key === "age" ? Math.round(v) : v;
  }
  if (!d.sex) errors.sex = "Выберите вариант";
  if (!d.goal) errors.goal = "Выберите цель";
  if (!d.level) errors.level = "Выберите опыт";
  if (!d.place) errors.place = "Выберите место";
  if (!d.activity) errors.activity = "Выберите активность";
  if (Object.keys(errors).length) return { errors };
  return {
    errors,
    body: {
      name: d.name.trim(),
      sex: d.sex!,
      ...nums,
      goal: d.goal!,
      level: d.level!,
      place: d.place!,
      activity: d.activity!,
      limitations: d.limitations.trim(),
    },
  };
}

export default function ProfilePage({ mode, onSaved, onCancel, onLogout }: ProfilePageProps) {
  const [draft, setDraft] = useState<Draft | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [errors, setErrors] = useState<Partial<Record<keyof Draft, string>>>({});
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = () => {
    setLoadError(null);
    myProfile()
      .then((p) => setDraft(toDraft(p)))
      .catch((e) => setLoadError(errorText(e)));
  };
  useEffect(load, []);

  const set = <K extends keyof Draft>(key: K, value: Draft[K]) => {
    setDraft((d) => (d ? { ...d, [key]: value } : d));
    setErrors((e) => ({ ...e, [key]: undefined }));
  };

  const submit = async (ev: FormEvent) => {
    ev.preventDefault();
    if (!draft || saving) return;
    const { body, errors: found } = validate(draft);
    setErrors(found);
    if (!body) {
      setSaveError("Заполните отмеченные поля");
      // Ошибка внизу длинной формы незаметна — показываем первое пустое поле.
      requestAnimationFrame(() =>
        document.querySelector(".profile-field.has-error")?.scrollIntoView({ block: "center", behavior: "smooth" }),
      );
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      onSaved(await saveProfile(body));
    } catch (e) {
      setSaveError(errorText(e));
      setSaving(false);
    }
  };

  let content: ReactNode;
  if (loadError) {
    content = (
      <div className="notice notice-error row">
        <span className="grow">Не удалось открыть анкету: {loadError}</span>
        <button type="button" className="btn btn-sm btn-outline" onClick={load}>
          Повторить
        </button>
      </div>
    );
  } else if (!draft) {
    content = <Spinner label="Открываем анкету" />;
  } else {
    content = (
      <form className="profile-form" onSubmit={submit} noValidate>
        <Field label="Как к вам обращаться" error={errors.name}>
          <input
            className="input"
            value={draft.name}
            maxLength={60}
            autoComplete="given-name"
            onChange={(e) => set("name", e.target.value)}
          />
        </Field>

        <Field label="Пол" error={errors.sex}>
          <Choice options={SEX} value={draft.sex} onChange={(v) => set("sex", v)} />
        </Field>

        <div className="profile-numbers">
          {(Object.keys(NUMBERS) as NumberKey[]).map((key) => (
            <Field key={key} label={NUMBERS[key].label} error={errors[key]}>
              <span className="profile-num">
                <input
                  className="input"
                  inputMode={key === "weight_kg" ? "decimal" : "numeric"}
                  value={draft[key]}
                  onChange={(e) => set(key, e.target.value)}
                />
                <span className="profile-unit">{NUMBERS[key].unit}</span>
              </span>
            </Field>
          ))}
        </div>

        <Field label="Цель" error={errors.goal}>
          <Choice options={GOAL} value={draft.goal} onChange={(v) => set("goal", v)} />
        </Field>

        <Field label="Опыт тренировок" error={errors.level}>
          <Choice options={LEVEL} value={draft.level} onChange={(v) => set("level", v)} />
        </Field>

        <Field label="Где тренируетесь" error={errors.place}>
          <Choice options={PLACE} value={draft.place} onChange={(v) => set("place", v)} />
        </Field>

        <Field label="Активность за неделю" error={errors.activity}>
          <Choice options={ACTIVITY} value={draft.activity} onChange={(v) => set("activity", v)} />
        </Field>

        <Field
          label="Травмы и ограничения"
          hint="Коуч учтёт их в ответах, а проверка программы не даст нагрузить больное место. Если нет — оставьте пустым."
        >
          <textarea
            className="textarea"
            value={draft.limitations}
            maxLength={400}
            placeholder="Например: болит колено при приседаниях"
            onChange={(e) => set("limitations", e.target.value)}
          />
        </Field>

        {saveError && <div className="notice notice-error">{saveError}</div>}

        <div className="profile-actions">
          <button type="submit" className="btn btn-primary btn-lg grow" disabled={saving}>
            {saving ? <Spinner size="sm" /> : mode === "first" ? "Готово, к коучу" : "Сохранить"}
          </button>
          {mode === "edit" && onCancel && (
            <button type="button" className="btn btn-lg" onClick={onCancel} disabled={saving}>
              Отмена
            </button>
          )}
        </div>
        {mode === "first" && (
          <p className="profile-foot">Анкету видит ваш тренер. Изменить её можно во вкладке «Программа».</p>
        )}
      </form>
    );
  }

  if (mode === "edit") {
    return (
      <div className="profile-page">
        <h2>Анкета</h2>
        <p className="muted small">
          По ней коуч считает вашу норму калорий и учитывает ограничения. Место тренировок меняет стартовую программу.
        </p>
        {content}
      </div>
    );
  }
  return (
    <div className="profile-first">
      <div className="profile-first-card">
        <div className="profile-first-head">
          <span className="brand-mark" aria-hidden="true">T</span>
          <h1 className="profile-first-title">Давайте познакомимся</h1>
          <p className="profile-first-lead">
            Минута — и коуч посчитает вашу норму калорий, а советы и программа учтут цель и травмы.
          </p>
        </div>
        {content}
        {onLogout && (
          <button type="button" className="btn btn-ghost btn-sm profile-out" onClick={onLogout}>
            Выйти
          </button>
        )}
      </div>
    </div>
  );
}

function Field({ label, hint, error, children }: { label: string; hint?: string; error?: string; children: ReactNode }) {
  return (
    <div className={`field profile-field ${error ? "has-error" : ""}`}>
      <span className="profile-label">{label}</span>
      {children}
      {error ? <span className="profile-error">{error}</span> : hint ? <span className="profile-hint">{hint}</span> : null}
    </div>
  );
}

function Choice<T extends string>({ options, value, onChange }: { options: Option<T>[]; value: T | null; onChange: (v: T) => void }) {
  return (
    <div className="profile-choice" role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={`profile-opt ${o.value === value ? "is-active" : ""}`}
          onClick={() => onChange(o.value)}
        >
          <span>{o.label}</span>
          {o.hint && <span className="profile-opt-hint">{o.hint}</span>}
        </button>
      ))}
    </div>
  );
}
