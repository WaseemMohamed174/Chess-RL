import { api } from "./api";

/* Shared by every page that offers "do it for all models" buttons. */

/** Start training for every model that is not training yet. Throws the capacity 409 when the
 *  machine cannot comfortably hold them (the page shows CapacityDialog, then calls again with force). */
export const startAllTraining = (force = false) =>
  api("/api/training/start-all", { method: "POST", body: { force } });

/** Delete the given models (all when ids is omitted). Models that are training are skipped. */
export const deleteModels = (ids) =>
  api("/api/models/bulk-delete", { method: "POST", body: { ids: ids ?? null } });

/** Address of the .zip with every model file (or just `ids`). */
export const downloadAllUrl = (ids) =>
  `/api/models/download-all${ids?.length ? `?ids=${ids.join(",")}` : ""}`;

/** Toasts for the answer of startAllTraining(). */
export function reportStart(r, t, toast) {
  if (r.started.length) toast(`${r.started.length} ${t("models started training")}`, "ok");
  if (r.failed.length) toast(`${r.failed.length} ${t("models could not start")}: ${r.failed.map((f) => `${f.name} (${f.reason})`).join(", ")}`, "error");
}

/** Toasts for the answer of deleteModels(). */
export function reportDelete(r, t, toast) {
  if (r.deleted.length) toast(`${r.deleted.length} ${t("models deleted")}`, "ok");
  if (r.skipped.length) toast(`${r.skipped.length} ${t("models were kept because they are training")}: ${r.skipped.map((s) => s.name).join(", ")}`, "error");
}
