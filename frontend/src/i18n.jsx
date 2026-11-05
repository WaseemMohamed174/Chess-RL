import { AR_TRAINING } from "./i18nTraining";
import { AR_PHASE5 } from "./i18nPhase5";
import { AR_POPULATION } from "./i18nPopulation";
import { AR_BULK } from "./i18nBulk";
import { createContext, useContext, useEffect, useMemo, useState } from "react";

// English text is the key. A missing Arabic entry simply falls back to English.
const AR_BASE = {
  Play: "العب", Spectate: "شاهد", Training: "التدريب", Models: "الموديلات",
  Architecture: "البنية", Settings: "الإعدادات",
  "Engine ready": "المحرك جاهز", "Engine starting": "المحرك يعمل...", "Engine offline": "المحرك متوقف",
  "New game": "لعبة جديدة", "Start game": "ابدأ اللعبة", Opponent: "الخصم", "Your color": "لونك",
  White: "أبيض", Black: "أسود", Random: "عشوائي", Undo: "تراجع", Hint: "تلميح", Resign: "استسلام",
  "Export PGN": "تصدير PGN", Moves: "النقلات", "What the model sees": "ما يراه الموديل",
  "Your move": "دورك", "Model is thinking": "الموديل يفكر", "You won": "فزت", "You lost": "خسرت", Draw: "تعادل",
  "Start match": "ابدأ المباراة", "Stop match": "أوقف المباراة", Games: "الألعاب", Speed: "السرعة",
  Simulations: "المحاكاة", Start: "ابدأ", Pause: "إيقاف مؤقت", Resume: "استئناف", Stop: "إيقاف",
  Iteration: "الدورة", "Games per minute": "ألعاب/دقيقة", Loss: "الخسارة", "Live self-play": "لعب ذاتي مباشر",
  Name: "الاسم", Size: "الحجم", Parameters: "البارامترز", Elo: "إيلو", Created: "أُنشئ",
  Rename: "إعادة تسمية", Clone: "نسخ", Download: "تنزيل", Delete: "حذف", "Create model": "أنشئ موديل",
  Save: "حفظ", "Reset to defaults": "الإعدادات الافتراضية", Language: "اللغة", "Board theme": "ثيم البورد",
  "Show arrows": "إظهار الأسهم", "Show evaluation": "إظهار التقييم", Interface: "الواجهة", Device: "الجهاز",
  Layers: "الطبقات", Input: "المدخل", "Policy head": "رأس السياسة", "Value head": "رأس القيمة",
  Search: "بحث", Cancel: "إلغاء", Saved: "تم الحفظ", "Coming soon": "قريباً", "Model A": "الموديل أ", "Model B": "الموديل ب", Game: "اللعبة", Wins: "انتصارات", Draws: "تعادلات", Slower: "أبطأ", Faster: "أسرع", "Match finished": "انتهت المباراة", "Start training": "ابدأ التدريب", "Games per iteration": "ألعاب كل دورة", Buffer: "الذاكرة", "Latest iteration": "آخر دورة", Promoted: "تمت الترقية", Kept: "بقي القديم", "Policy loss": "خسارة السياسة", "Value loss": "خسارة القيمة", Throughput: "السرعة", Tiny: "صغير جدًا", Small: "صغير", Medium: "متوسط", Custom: "مخصص", Blocks: "البلوكات", Filters: "الفلاتر", "Size on disk": "الحجم على القرص", Operations: "العمليات", "All layers": "كل الطبقات", Actions: "إجراءات", Type: "النوع", "Output shape": "شكل المخرج", "Tree search": "بحث الشجرة", Network: "الشبكة", "Saved changes": "اتحفظت التعديلات", "Unsaved changes": "تعديلات غير محفوظة", Discard: "تراجع عن التعديل", Memory: "الذاكرة", Processor: "المعالج",
 "Game time": "وقت اللعبة", "This move": "هذه النقلة", "Match time": "وقت المباراة", Checkmate: "كش مات", "The white king is dead": "الملك الأبيض مات", "The black king is dead": "الملك الأسود مات", Close: "إغلاق", Sound: "الصوت",
 "Take back your last move (and the model's reply).": "ارجع عن نقلتك الأخيرة (ورد الموديل عليها).", "Ask the model what it would play for you. It is drawn as a green arrow.": "اسأل الموديل عن النقلة اللي كان هيلعبها مكانك. بتظهر كسهم أخضر.", "Give up this game. It counts as a loss.": "استسلم في اللعبة دي. بتتحسب خسارة.", "Download this game as a PGN file, which any chess program can open.": "نزّل اللعبة كملف PGN تفتحه أي برنامج شطرنج.", "Every move of this game, White first.": "كل نقلات اللعبة دي، الأبيض الأول.",
 "Model": "موديل", Human: "إنسان", Player: "لاعب", "Your turn": "دورك", "Thinking...": "بيفكر...", "Viewing an earlier move": "بتشوف نقلة سابقة", "Back to current move": "ارجع للنقلة الحالية", "Click a move to see the board at that moment. Small numbers are how long each move took.": "اضغط على أي نقلة تشوف الرقعة في اللحظة دي. الأرقام الصغيرة هي الوقت اللي أخدته كل نقلة.", "Plays the light pieces": "بيلعب بالقطع البيضاء", "Plays the dark pieces": "بيلعب بالقطع السودا",
};

const AR = { ...AR_BASE, ...AR_TRAINING, ...AR_PHASE5, ...AR_POPULATION, ...AR_BULK };

const Ctx = createContext({ lang: "en", setLang: () => {}, t: (s) => s });

export function I18nProvider({ children }) {
  const [lang, setLang] = useState(() => localStorage.getItem("lang") || "en");
  useEffect(() => {
    localStorage.setItem("lang", lang);
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  }, [lang]);
  const value = useMemo(
    () => ({ lang, setLang, t: (s) => (lang === "ar" && AR[s]) || s }),
    [lang],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useI18n = () => useContext(Ctx);
