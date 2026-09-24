interface SpinnerProps {
  size?: "sm" | "md" | "lg";
  /** Подпись рядом со спиннером; если есть — рендерится блок загрузки. */
  label?: string;
  className?: string;
}

/** Кольцевой спиннер. С label — центрированный блок «спиннер + текст». */
export default function Spinner({ size = "md", label, className = "" }: SpinnerProps) {
  const ring = (
    <span
      className={`spinner ${size === "sm" ? "spinner-sm" : size === "lg" ? "spinner-lg" : ""} ${className}`}
      role="status"
      aria-label={label ?? "Загрузка"}
    />
  );
  if (!label) return ring;
  return (
    <div className="loading-block">
      {ring}
      <span>{label}</span>
    </div>
  );
}
