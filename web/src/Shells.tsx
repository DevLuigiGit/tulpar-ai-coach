// Оболочки по ролям: шапка, вкладки и роутинг по hash. Экраны получают только
// данные и колбэки навигации — App и Shells агенты экранов не трогают.
import { useEffect, useState, type ReactNode } from "react";
import { queue } from "./api";
import type { User } from "./types";
import { initials } from "./format";
import { navigate, useRoute } from "./route";
import ChatPage from "./pages/client/ChatPage";
import PlanPage from "./pages/client/PlanPage";
import QueuePage from "./pages/trainer/QueuePage";
import ClientsPage from "./pages/trainer/ClientsPage";
import HistoryPage from "./pages/trainer/HistoryPage";

interface ShellProps {
  user: User;
  onLogout: () => void;
}

interface TabDef {
  id: string;
  label: string;
  badge?: number;
}

function Header({ user, onLogout, sub }: ShellProps & { sub: string }) {
  return (
    <header className="app-header">
      <div className="app-header-inner">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">T</span>
          <span>
            Tulpar <span className="brand-sub">{sub}</span>
          </span>
        </div>
        <div className="grow" />
        <div className="user-pill">
          <span className="avatar" aria-hidden="true">{initials(user.name)}</span>
          <span className="user-name truncate">{user.name}</span>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={onLogout}>
          Выйти
        </button>
      </div>
    </header>
  );
}

function Tabs({ tabs, active }: { tabs: TabDef[]; active: string }) {
  return (
    <nav className="app-nav">
      <div className="tabs" role="tablist">
        {tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={t.id === active}
            className={`tab ${t.id === active ? "is-active" : ""}`}
            onClick={() => navigate(t.id)}
          >
            {t.label}
            {!!t.badge && <span className="tab-badge">{t.badge}</span>}
          </button>
        ))}
      </div>
    </nav>
  );
}

function Frame(props: { kind: "client" | "trainer"; header: ReactNode; tabs: ReactNode; fill: boolean; children: ReactNode }) {
  return (
    <div className={`app shell-${props.kind}`}>
      {props.header}
      {props.tabs}
      <main className={`app-main ${props.fill ? "is-fill" : ""}`}>{props.children}</main>
    </div>
  );
}

// ---------- Клиент: «Коуч» и «Программа» ----------

const CLIENT_TABS: TabDef[] = [
  { id: "coach", label: "Коуч" },
  { id: "plan", label: "Программа" },
];

export function ClientShell({ user, onLogout }: ShellProps) {
  const route = useRoute();
  const tab = CLIENT_TABS.some((t) => t.id === route.tab) ? route.tab : "coach";
  return (
    <Frame
      kind="client"
      header={<Header user={user} onLogout={onLogout} sub="AI Coach" />}
      tabs={<Tabs tabs={CLIENT_TABS} active={tab} />}
      fill={tab === "coach"}
    >
      {tab === "coach" ? (
        <ChatPage user={user} onOpenPlan={() => navigate("plan")} />
      ) : (
        <PlanPage user={user} onOpenChat={() => navigate("coach")} />
      )}
    </Frame>
  );
}

// ---------- Тренер: «Очередь», «Клиенты», «История» ----------

export function TrainerShell({ user, onLogout }: ShellProps) {
  const route = useRoute();
  const [pending, setPending] = useState(0);

  // Счётчик на вкладке «Очередь»: ждут решения + открытые эскалации. Обновляем раз в 20 с;
  // QueuePage может сообщить число сразу через onCountChange.
  useEffect(() => {
    let alive = true;
    const tick = () =>
      queue(false)
        .then((items) => {
          if (alive) setPending(items.filter((p) => p.status === "pending" || p.status === "open").length);
        })
        .catch(() => {});
    tick();
    const id = window.setInterval(tick, 20_000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);
  const tabs: TabDef[] = [
    { id: "queue", label: "Очередь", badge: pending },
    { id: "clients", label: "Клиенты" },
    { id: "history", label: "История" },
  ];
  const tab = tabs.some((t) => t.id === route.tab) ? route.tab : "queue";
  const openClient = (id: string) => navigate("clients", id);

  return (
    <Frame
      kind="trainer"
      header={<Header user={user} onLogout={onLogout} sub="Кабинет тренера" />}
      tabs={<Tabs tabs={tabs} active={tab} />}
      fill={false}
    >
      {tab === "queue" && <QueuePage user={user} onOpenClient={openClient} onCountChange={setPending} />}
      {tab === "clients" && (
        <ClientsPage
          user={user}
          selectedClientId={route.param}
          onSelectClient={(id) => navigate("clients", id)}
        />
      )}
      {tab === "history" && <HistoryPage user={user} onOpenClient={openClient} />}
    </Frame>
  );
}
