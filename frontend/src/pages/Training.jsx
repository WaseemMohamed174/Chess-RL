import { useState } from "react";
import PageHead from "../components/PageHead";
import PopulationPanel from "../components/PopulationPanel";
import TrainingRuns from "../components/TrainingRuns";
import { Segmented } from "../components/ui";
import { useI18n } from "../i18n";

/** Two views of the same page: single models training, and populations of models. */
export default function Training() {
  const { t } = useI18n();
  const [view, setView] = useState(() => (location.hash === "#population" ? "population" : "runs"));
  const pick = (v) => {
    setView(v);
    history.replaceState(null, "", v === "population" ? "#population" : location.pathname);
  };
  return (
    <>
      <div style={{ marginBottom: 18 }}>
        <Segmented label={t("Training view")} value={view} onChange={pick} options={[
          { value: "runs", label: t("Single models") },
          { value: "population", label: t("Population") }]} />
      </div>
      {view === "population" ? (
        <>
          <PageHead title="Population"
            subtitle="A group of models trains together. The strongest are kept, the rest are replaced by slightly changed copies of them, generation after generation." />
          <PopulationPanel />
        </>
      ) : <TrainingRuns scope="single" />}
    </>
  );
}
