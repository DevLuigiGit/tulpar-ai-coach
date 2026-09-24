// Мини-роутер на location.hash: вкладка переживает перезагрузку, а FastAPI
// не нужно знать о клиентских путях. Формат: #/<tab>[/<param>].
import { useEffect, useState } from "react";

export interface Route {
  tab: string;
  param: string | null;
}

export function parseHash(hash = window.location.hash): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  return { tab: parts[0] ?? "", param: parts[1] ? decodeURIComponent(parts[1]) : null };
}

export function navigate(tab: string, param?: string | null): void {
  const next = `#/${tab}${param ? `/${encodeURIComponent(param)}` : ""}`;
  if (window.location.hash !== next) window.location.hash = next;
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash());
  useEffect(() => {
    const on = () => setRoute(parseHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
