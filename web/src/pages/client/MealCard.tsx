/**
 * MealCard — карточка распознанного приёма пищи в чате.
 *
 * Props:
 *   card: MealCardData        — ChatReply.meal (card_id, items, unknown). КБЖУ позиций — на 100 г.
 *   initiallyLogged?: boolean — карточка уже записана (по истории: за ней идёт заметка meal_logged).
 *   onLogged?: (r) => void    — после успешного confirmMeal.
 *
 * Граммы редактируются по позициям, КБЖУ порции пересчитывается сразу (value * grams / 100).
 * «Записать» → POST /api/meals/{card_id}/confirm { grams: {"0": g, ...}, meal }.
 */
import { useRef, useState } from "react";
import { confirmMeal, errorText } from "../../api";
import type { ConfirmMealResponse, MealCard as MealCardData, MealSlot } from "../../types";
import { MEAL_LABEL, fmt0, fmtKcal, mealByTime, portion } from "../../format";
import Spinner from "../../components/Spinner";
import { CheckIcon } from "./icons";
import { MealItemRow, MealUnknown } from "./MealParts";
import "./client.css";
import "./meal.css";

export interface MealCardProps {
  card: MealCardData;
  initiallyLogged?: boolean;
  onLogged?: (r: ConfirmMealResponse) => void;
}

const SLOTS: MealSlot[] = ["breakfast", "lunch", "dinner", "snack"];
export const MAX_GRAMS = 3000;

export function parseGrams(v: string): number {
  const n = Number(v.replace(",", "."));
  return v.trim() !== "" && Number.isFinite(n) && n >= 0 && n <= MAX_GRAMS ? n : NaN;
}

export default function MealCard({ card, initiallyLogged = false, onLogged }: MealCardProps) {
  const [grams, setGrams] = useState<string[]>(() => card.items.map((i) => String(Math.round(i.grams))));
  const [touched, setTouched] = useState<boolean[]>(() => card.items.map(() => false));
  const [meal, setMeal] = useState<MealSlot>(() => mealByTime());
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<ConfirmMealResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);

  const parsed = grams.map(parseGrams);
  const allValid = parsed.every((g) => !Number.isNaN(g));
  const totals = { kcal: 0, protein: 0, fat: 0, carbs: 0 };
  card.items.forEach((item, i) => {
    const g = parsed[i];
    if (Number.isNaN(g)) return;
    const p = portion(item, g);
    totals.kcal += p.kcal;
    totals.protein += p.protein;
    totals.fat += p.fat;
    totals.carbs += p.carbs;
  });

  const done = !!result || initiallyLogged;
  const canSave = !done && !saving && allValid && parsed.some((g) => g > 0);

  const setAt = (i: number, v: string) => {
    setGrams((g) => g.map((x, j) => (j === i ? v : x)));
    setTouched((t) => t.map((x, j) => (j === i ? true : x)));
    setError(null);
  };

  const save = async () => {
    if (!canSave || inFlight.current) return;
    inFlight.current = true;
    setSaving(true);
    setError(null);
    const body: Record<string, number> = {};
    parsed.forEach((g, i) => (body[String(i)] = g));
    try {
      const r = await confirmMeal(card.card_id, { grams: body, meal });
      setResult(r);
      onLogged?.(r);
    } catch (e) {
      setError(errorText(e));
    } finally {
      inFlight.current = false;
      setSaving(false);
    }
  };

  return (
    <div className={`meal ${done ? "is-done" : ""}`}>
      <div className="meal-head">
        <div className="stack tight" style={{ gap: 2 }}>
          <span className="label">Приём пищи</span>
          <span className="meal-total">
            <span className="num">{fmtKcal(totals.kcal)}</span> <span className="meal-total-unit">ккал</span>
          </span>
        </div>
        <div className="meal-macros">
          <Macro name="Б" value={totals.protein} />
          <Macro name="Ж" value={totals.fat} />
          <Macro name="У" value={totals.carbs} />
        </div>
      </div>

      {card.items.length > 0 ? (
        <ul className="meal-items">
          {card.items.map((item, i) => (
            <MealItemRow
              key={`${item.food_id}-${i}`}
              item={item}
              value={grams[i]}
              grams={parsed[i]}
              askGrams={item.grams_source === "default" && !touched[i]}
              disabled={done || saving}
              onChange={(v) => setAt(i, v)}
            />
          ))}
        </ul>
      ) : (
        <div className="meal-empty muted small">Не нашёл продукты в справочнике — уточните название.</div>
      )}

      {card.unknown.length > 0 && <MealUnknown items={card.unknown} />}

      {done ? (
        <div className="meal-done" role="status">
          <CheckIcon size={16} />
          {result ? (
            <span>
              {result.already ? "Уже записано" : "Записано"} · <span className="num">{fmt0(result.total_kcal)}</span> ккал
            </span>
          ) : (
            <span>Записано в дневник</span>
          )}
        </div>
      ) : (
        card.items.length > 0 && (
          <div className="meal-actions">
            <div className="meal-slots" role="radiogroup" aria-label="Приём пищи">
              {SLOTS.map((s) => (
                <button
                  key={s}
                  type="button"
                  role="radio"
                  aria-checked={meal === s}
                  className={`meal-slot ${meal === s ? "is-active" : ""}`}
                  disabled={saving}
                  onClick={() => setMeal(s)}
                >
                  {MEAL_LABEL[s]}
                </button>
              ))}
            </div>
            <button className="btn btn-primary btn-block" disabled={!canSave} onClick={() => void save()}>
              {saving ? <Spinner size="sm" /> : null}
              {saving ? "Записываем…" : "Записать"}
            </button>
            {!allValid && <div className="meal-warn">Граммы — число от 0 до {MAX_GRAMS}</div>}
          </div>
        )
      )}

      {error && <div className="notice notice-error meal-error">{error}</div>}
    </div>
  );
}

function Macro({ name, value }: { name: string; value: number }) {
  return (
    <span className="meal-macro">
      <span className="meal-macro-name">{name}</span>
      <span className="num">{fmt0(value)}</span>
    </span>
  );
}
