// Типы контракта API Tulpar AI Coach. Держим один файл, чтобы клиентские
// и тренерские экраны опирались на одинаковые формы данных.

export type Role = "client" | "trainer";

export interface User {
  id: string;
  role: Role;
  name: string;
}

export interface LoginResponse {
  token: string;
  user: User;
}

// ---------- Чат клиента ----------

export type ReplyKind =
  | "answer"
  | "meal_card"
  | "escalated"
  | "proposal"
  | "refusal"
  | "info";

export type CitationSource = "exercises" | "nutrition" | "who2020";

export interface Citation {
  n: number;
  title: string;
  page: number | null;
  source: CitationSource;
}

/** Значения kcal/protein/fat/carbs даны на 100 г; порция = value * grams / 100. */
export interface MealItem {
  food_id: string;
  name: string;
  grams: number;
  kcal: number;
  protein: number;
  fat: number;
  carbs: number;
  asked_as: string;
  grams_source: "user" | "default";
}

export interface UnknownFood {
  name: string;
  alternatives: string[];
}

export interface MealCard {
  card_id: string;
  items: MealItem[];
  unknown: UnknownFood[];
}

export interface ChatReply {
  reply: string;
  kind: ReplyKind;
  intent: string | null;
  citations: Citation[];
  meal: MealCard | null;
  proposal_id: string | null;
  escalation_id: string | null;
  transcript: string | null;
}

/** payload сообщения ассистента: поля ChatReply без reply. */
export type ReplyPayload = Omit<ChatReply, "reply">;

/** Служебные payload: запись еды в дневник и ответ тренера в чат. */
export interface SystemPayload {
  kind: "meal_logged" | "trainer_reply";
  [key: string]: unknown;
}

/** payload сообщения пользователя: сервер отмечает вложения. */
export interface UserPayload {
  has_photo?: boolean;
  has_audio?: boolean;
}

export type MessagePayload = ReplyPayload | SystemPayload | UserPayload;

export interface ChatMessage {
  /** Сервер отдаёт целое (autoincrement), локальные сообщения — строку "local-…". */
  id: number | string;
  client_id: string;
  role: "user" | "assistant";
  text: string;
  payload: MessagePayload | null;
  created_at: string;
}

export type MealSlot = "breakfast" | "lunch" | "dinner" | "snack";

export interface ConfirmMealRequest {
  /** Ключ — индекс позиции в MealCard.items (строкой), значение — граммы. */
  grams: Record<string, number>;
  meal: MealSlot;
}

export interface ConfirmMealResponse {
  logged: number;
  already: boolean;
  total_kcal: number;
}

// ---------- Программа ----------

export interface PlanExercise {
  id: string;
  exercise_id: string;
  exercise_name: string;
  muscle_group: string | null;
  equipment: string | null;
  target_sets: number;
  target_reps: string | number;
}

export interface PlanDay {
  id: string;
  day_index: number;
  title: string;
  weekday: number | string | null;
  exercises: PlanExercise[];
}

export interface Plan {
  id: string;
  title: string;
  days: PlanDay[];
  /** Необязательные поля каталога программ (fixtures/plans.json). */
  mode?: string;
  meta?: { subtitle?: string; days_per_week?: number; [key: string]: unknown };
}

// ---------- Предложения и эскалации ----------

export type ProposalKind = "program" | "escalation";

export type ProposalStatus =
  | "drafting"
  | "pending"
  | "applied"
  | "rejected"
  | "failed"
  | "open"
  | "resolved";

/** Операция правки программы; форма зависит от op, поэтому держим открытой. */
export interface ProgramOp {
  op: string;
  [key: string]: unknown;
}

export interface ProgramDraft {
  summary: string;
  rationale: string;
  ops: ProgramOp[];
}

export interface EscalationDraft {
  reason: string;
  intent: string;
}

export interface Violation {
  code: string;
  severity: "error" | "warning";
  message: string;
  op_index: number | null;
}

export interface Change {
  op: string;
  day: string;
  was: string | null;
  becomes: string;
  reason: string;
}

export type DecisionAction = "accept" | "reject" | "edit";

export interface Decision {
  action: DecisionAction;
  comment: string | null;
}

export interface Proposal {
  id: string;
  kind: ProposalKind;
  client_id: string;
  client_name: string;
  trainer_id: string;
  source: "client" | "trainer";
  request: string;
  status: ProposalStatus;
  draft: ProgramDraft | EscalationDraft | null;
  violations: Violation[] | null;
  /** Только у kind="program"; у эскалаций поля нет. */
  changes?: Change[] | null;
  before: Plan | null;
  after: Plan | null;
  decision: Decision | null;
  reply: string | null;
  created_at: string;
  updated_at: string;
}

export function isProgramDraft(d: Proposal["draft"]): d is ProgramDraft {
  return !!d && "ops" in d;
}

export function isEscalationDraft(d: Proposal["draft"]): d is EscalationDraft {
  return !!d && "reason" in d;
}

// ---------- Тренер: клиенты ----------

export interface ClientSummary {
  id: string;
  name: string;
  goal: string;
  level: string;
  place: string;
  injury: boolean;
  active_plan_title: string | null;
}

export interface ClientProfile {
  sex: string;
  age: number | null;
  height_cm: number | null;
  goal: string;
  level: string;
  place: string;
  training_days: number[];
  activity: string;
  weight_kg: number | null;
  goal_weight_kg: number | null;
}

export interface TrainerNote {
  injury: boolean;
  body: string;
}

/** Если записей нет, сервер отдаёт только {"days_logged": 0} — поля avg_* отсутствуют. */
export interface Nutrition14d {
  days_logged: number;
  avg_kcal?: number | null;
  avg_protein?: number | null;
  avg_fat?: number | null;
  avg_carbs?: number | null;
}

export interface RecentSession {
  completed_at: string;
  day_title: string;
  avg_rpe: number | null;
}

export interface WeightPoint {
  measured_at: string;
  weight_kg: number;
}

export interface ClientContext {
  client_id: string;
  name: string;
  profile: ClientProfile;
  trainer_note: TrainerNote | null;
  active_plan: Plan | null;
  nutrition_14d: Nutrition14d;
  recent_sessions: RecentSession[];
  weights: WeightPoint[];
}
