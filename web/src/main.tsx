import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { startTelegram } from "./telegram";
import "./styles.css";

// Внутри Telegram: класс tg-webapp, запрет масштаба и высота видимой области — до первого кадра,
// чтобы вёрстка не прыгала. Повторный вызов из App ничего не делает; вне Telegram — тоже ничего.
startTelegram();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
