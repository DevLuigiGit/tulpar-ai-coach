// Строки карточки еды: позиция с редактируемыми граммами и список «нет в справочнике».
import { useId } from "react";
import type { MealItem, UnknownFood } from "../../types";
import { fmt0, fmtKcal, portion } from "../../format";

interface RowProps {
  item: MealItem;
  /** Сырое значение поля ввода. */
  value: string;
  /** Разобранные граммы; NaN — поле заполнено неверно. */
  grams: number;
  /** Граммы подставлены по умолчанию — просим уточнить. */
  askGrams: boolean;
  disabled: boolean;
  onChange: (v: string) => void;
}

export function MealItemRow({ item, value, grams, askGrams, disabled, onChange }: RowProps) {
  const invalid = Number.isNaN(grams);
  const p = portion(item, invalid ? 0 : grams);
  const asked = item.asked_as && item.asked_as.toLowerCase() !== item.name.toLowerCase() ? item.asked_as : null;
  const inputId = useId();

  return (
    <li className={`meal-item ${grams === 0 ? "is-zero" : ""}`}>
      <div className="meal-item-main">
        <label className="meal-item-name" htmlFor={inputId}>
          {item.name}
        </label>
        {asked && <div className="meal-item-asked truncate">вы написали: «{asked}»</div>}
        <div className="meal-item-kbju num">
          <span className="meal-item-kcal">{invalid ? "—" : fmtKcal(p.kcal)} ккал</span>
          <span>Б {invalid ? "—" : fmt0(p.protein)}</span>
          <span>Ж {invalid ? "—" : fmt0(p.fat)}</span>
          <span>У {invalid ? "—" : fmt0(p.carbs)}</span>
        </div>
        {askGrams && <div className="meal-hint">уточните граммы</div>}
      </div>
      <div className={`meal-grams ${invalid ? "is-invalid" : ""} ${askGrams ? "is-ask" : ""}`}>
        <input
          id={inputId}
          className="meal-grams-input num"
          type="number"
          inputMode="decimal"
          min={0}
          max={3000}
          step={10}
          value={value}
          disabled={disabled}
          aria-invalid={invalid}
          aria-label={`Граммы: ${item.name}`}
          onChange={(e) => onChange(e.target.value)}
          onFocus={(e) => e.target.select()}
        />
        <span className="meal-grams-unit">г</span>
      </div>
    </li>
  );
}

export function MealUnknown({ items }: { items: UnknownFood[] }) {
  return (
    <div className="meal-unknown">
      {items.map((u) => (
        <div key={u.name} className="meal-unknown-row">
          <div className="meal-unknown-head">
            <span className="meal-unknown-name">{u.name}</span>
            <span className="meal-unknown-tag">нет в справочнике</span>
          </div>
          {u.alternatives.length > 0 && (
            <div className="meal-unknown-alt">Похожее: {u.alternatives.slice(0, 4).join(", ")}</div>
          )}
        </div>
      ))}
    </div>
  );
}
