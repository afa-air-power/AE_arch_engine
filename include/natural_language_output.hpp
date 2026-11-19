#pragma once
// ============================================================================
// AE Arch Engine (AE::mind_v1.1_Stable)
// Module 6.1: Natural-Language Verbalizer + Parser Round-Trip Verification
// Author: Andrey Firstov (IU5 BMSTU)
//
// Выход движка формируется на естественном языке из интент-вектора и
// морфемной маски. Сгенерированный текст сразу возвращается в языковой парсер
// (ru/en), повторно разбирается на морфемы и сравнивается с ожидаемой маской.
// Если семантические кластеры потерялись, вербализатор детерминированно
// дополняет предложение и повторяет разбор — это и есть замкнутый контур
// "интент -> текст -> парсер -> проверка".
// ============================================================================

#include "lang/parser.hpp"
#include "morphemic_trie.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ae::mind {

// Отчёт обратного разбора: что именно парсер увидел в сгенерированном тексте.
struct NaturalLanguageReport {
    std::string text;              // ответ на естественном языке
    std::string feedback;          // человекочитаемый итог сверки с парсером
    std::uint64_t expected_clusters = 0;
    std::uint64_t parsed_clusters = 0;
    double cluster_recall = 1.0;   // доля ожидаемых кластеров, найденных парсером
    double cluster_precision = 0.0; // доля найденных кластеров, бывших в ожидании
    int reparsed_morphemes = 0;    // морфем в тексте по данным языкового парсера
    int refinement_passes = 0;     // сколько раз текст пересобирался и переразбирался
    bool consistent = false;       // контур сошёлся: не потеряли смысл и не добавили лишнего
    lang::ParseResult reparsed;    // полный результат обратного разбора
};

namespace nl_detail {

// Биты семантических кластеров [28..47] — смысловое ядро маски.
// Типы морфем [24..27] и MORPH-кластер (бит 48) в сверке не участвуют:
// служебные окончания не должны считаться потерей смысла.
inline constexpr std::uint64_t CLUSTER_WINDOW =
    ((std::uint64_t{1} << 20) - 1) << 28;

struct DomainLexeme {
    MorphemicPatriciaTrie::Semantic cluster;
    std::string_view ru;
    std::string_view en;
    std::string_view def_ru;  // предикат того же кластера: "граф — это <def_ru>"
    std::string_view def_en;
};

// Индекс совпадает с Semantic (0..19), поэтому intent[0..15] напрямую
// отображается в домены 0..15 из README (GRAPH..LAW).
// Второе слово определения взято из того же кластера, поэтому предикат
// не добавляет новых доменов и не ломает точность обратной сверки.
inline constexpr std::array<DomainLexeme, 20> kDomainLexemes{{
    {MorphemicPatriciaTrie::Semantic::GRAPH,   "граф",          "graph",     "структура",  "structure"},
    {MorphemicPatriciaTrie::Semantic::KNOW,    "знание",        "knowledge", "смысл",      "meaning"},
    {MorphemicPatriciaTrie::Semantic::FLOW,    "поток",         "flow",      "процесс",    "process"},
    {MorphemicPatriciaTrie::Semantic::TEXT,    "текст",         "text",      "язык",       "symbol"},
    {MorphemicPatriciaTrie::Semantic::NUM,     "число",         "number",    "вычисление", "metric"},
    {MorphemicPatriciaTrie::Semantic::TIME,    "время",         "time",      "интервал",   "period"},
    {MorphemicPatriciaTrie::Semantic::SPACE,   "пространство",  "space",     "координаты", "point"},
    {MorphemicPatriciaTrie::Semantic::MIND,    "нейрон",        "brain",     "мозг",       "behavior"},
    {MorphemicPatriciaTrie::Semantic::COMPUTE, "программа",     "program",   "код",        "code"},
    {MorphemicPatriciaTrie::Semantic::NET,     "сеть",          "internet",  "протокол",   "protocol"},
    {MorphemicPatriciaTrie::Semantic::PHYS,    "физика",        "physics",   "энергия",    "energy"},
    {MorphemicPatriciaTrie::Semantic::CHEM,    "химия",         "chemistry", "реакция",    "reaction"},
    {MorphemicPatriciaTrie::Semantic::BIO,     "биология",      "biology",   "клетка",     "cell"},
    {MorphemicPatriciaTrie::Semantic::MED,     "медицина",      "medicine",  "лечение",    "treatment"},
    {MorphemicPatriciaTrie::Semantic::ECON,    "экономика",     "economy",   "рынок",      "market"},
    {MorphemicPatriciaTrie::Semantic::LAW,     "закон",         "law",       "право",      "contract"},
    {MorphemicPatriciaTrie::Semantic::SOC,     "общество",      "society",   "культура",   "culture"},
    {MorphemicPatriciaTrie::Semantic::QUALITY, "качество",      "quality",   "свойство",   "property"},
    {MorphemicPatriciaTrie::Semantic::NATURE,  "природа",       "climate",   "климат",     "weather"},
    {MorphemicPatriciaTrie::Semantic::ACTION,  "создание",      "transform", "анализ",     "build"},
}};

inline std::uint64_t cluster_bits(std::uint64_t mask) noexcept {
    return mask & CLUSTER_WINDOW;
}

inline bool is_russian(std::string_view text) noexcept {
    return lang::detect_language(text) == lang::Language::RUSSIAN;
}

inline std::string_view domain_word(std::size_t idx, bool ru) noexcept {
    const DomainLexeme& lex = kDomainLexemes[idx];
    return ru ? lex.ru : lex.en;
}

inline std::string_view definition_word(std::size_t idx, bool ru) noexcept {
    const DomainLexeme& lex = kDomainLexemes[idx];
    return ru ? lex.def_ru : lex.def_en;
}

inline std::string join_domains(const std::vector<int>& domains, bool ru) {
    std::string out;
    for (std::size_t i = 0; i < domains.size(); ++i) {
        if (i > 0) {
            if (i + 1 == domains.size()) {
                out += ru ? " и " : " and ";
            } else {
                out += ", ";
            }
        }
        out += domain_word(static_cast<std::size_t>(domains[i]), ru);
    }
    return out;
}

// Ответ состоит из двух частей: перечисление подтверждённых доменов и
// определения-предикаты. Каждое определение использует второе слово из того же
// кластера, поэтому грамматика богаче, а маска не приобретает лишних доменов.
inline std::string render_text(const std::vector<int>& domains,
                               int max_flow,
                               int bottleneck,
                               bool ru) {
    std::string text;
    text.reserve(160);
    if (domains.empty()) {
        text = ru ? "Тема: нет." : "Topic: none.";
    } else {
        text = ru ? "Тема: " : "Topic: ";
        text += join_domains(domains, ru);
        text += '.';

        const std::size_t definitions =
            std::min<std::size_t>(2, domains.size());
        text += ru ? " Уточнение: " : " Detail: ";
        for (std::size_t i = 0; i < definitions; ++i) {
            if (i > 0) text += ru ? ", а " : ", and ";
            const std::size_t idx = static_cast<std::size_t>(domains[i]);
            text += domain_word(idx, ru);
            text += ru ? " — это " : " is ";
            text += definition_word(idx, ru);
        }
        text += '.';
    }
    text += ' ';
    text += std::to_string(max_flow);
    text += '/';
    text += std::to_string(bottleneck);
    return text;
}

struct RoundTrip {
    std::uint64_t clusters = 0;
    int morphemes = 0;
    lang::ParseResult parsed;
};

// Ключевая операция: сгенерированный текст возвращается в тот же языковой
// парсер, которым разбирался запрос, и в морфемный триe движка.
inline RoundTrip parse_back(std::string_view text,
                            const MorphemicPatriciaTrie& trie) {
    RoundTrip out;
    if (lang::LanguageInterface* parser = lang::select_parser(text)) {
        out.parsed = parser->parse(text);
        out.morphemes = static_cast<int>(out.parsed.morphemes.size());
    }
    const std::array<float, 16> neutral_intent{};
    auto [mask, count] = trie.compile_morpheme_mask(text, neutral_intent);
    out.clusters = cluster_bits(mask);
    if (out.morphemes == 0) {
        out.morphemes = count;
    }
    return out;
}

inline std::string describe(std::string_view text, const RoundTrip& trip,
                            double recall, double precision, int passes,
                            bool ru) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    if (ru) {
        oss << "Обратный разбор: язык="
            << (trip.parsed.language == lang::Language::RUSSIAN ? "ru" : "en")
            << ", морфем=" << trip.morphemes
            << ", полнота=" << recall
            << ", точность=" << precision
            << ", проходов=" << passes
            << ". Текст: \"" << text << "\".";
    } else {
        oss << "Round-trip: lang="
            << (trip.parsed.language == lang::Language::RUSSIAN ? "ru" : "en")
            << ", morphemes=" << trip.morphemes
            << ", recall=" << recall
            << ", precision=" << precision
            << ", passes=" << passes
            << ". Text: \"" << text << "\".";
    }
    return oss.str();
}

} // namespace nl_detail

// Строит ответ на естественном языке из интент-вектора и ожидаемой маски,
// затем до трёх раз переразбирает его и дополняет потерянные домены.
inline NaturalLanguageReport render_natural_language(
    std::string_view query,
    const std::array<float, 16>& intent,
    std::uint64_t expected_mask,
    int max_flow,
    int bottleneck,
    const MorphemicPatriciaTrie& trie) {
    using namespace nl_detail;

    NaturalLanguageReport report;
    const bool ru = is_russian(query);
    report.expected_clusters = cluster_bits(expected_mask);

    // 1. Домены, реально подтверждённые морфемной маской запроса.
    std::vector<int> present;
    present.reserve(kDomainLexemes.size());
    for (std::size_t i = 0; i < kDomainLexemes.size(); ++i) {
        if (report.expected_clusters &
            MorphemicPatriciaTrie::cluster_mask(kDomainLexemes[i].cluster)) {
            present.push_back(static_cast<int>(i));
        }
    }

    // 2. Порядок: самый сильный интент первым, затем по индексу домена.
    std::stable_sort(present.begin(), present.end(), [&](int a, int b) {
        const float sa = a < static_cast<int>(intent.size()) ? intent[a] : -2.0f;
        const float sb = b < static_cast<int>(intent.size()) ? intent[b] : -2.0f;
        return sa > sb;
    });

    // 3. Контур "текст -> парсер -> сверка". Сначала пробуем компактный ответ,
    //    затем добавляем недостающие домены, пока полнота не станет полной.
    const std::size_t initial = std::min<std::size_t>(3, present.size());

    double best_score = -1.0;
    for (std::size_t take = initial; take <= present.size(); ++take) {
        std::vector<int> chosen(present.begin(), present.begin() + take);
        std::string text = render_text(chosen, max_flow, bottleneck, ru);
        RoundTrip trip = parse_back(text, trie);

        const double expected_n =
            static_cast<double>(std::popcount(report.expected_clusters));
        const double parsed_n =
            static_cast<double>(std::popcount(trip.clusters));
        const double both_n = static_cast<double>(
            std::popcount(report.expected_clusters & trip.clusters));

        const double recall = expected_n == 0.0 ? 1.0 : both_n / expected_n;
        const double precision = parsed_n == 0.0 ? 1.0 : both_n / parsed_n;
        const double score = recall + precision;

        if (score > best_score) {
            best_score = score;
            report.text = std::move(text);
            report.parsed_clusters = trip.clusters;
            report.cluster_recall = recall;
            report.cluster_precision = precision;
            report.reparsed_morphemes = trip.morphemes;
            report.refinement_passes = static_cast<int>(take - initial);
            report.reparsed = std::move(trip.parsed);
        }
        if (recall >= 0.999) break;
    }

    // 4. Порог сходимости: смысл не потерян и не разбавлен лишними доменами.
    report.consistent = report.cluster_recall >= 0.99 &&
                        report.cluster_precision >= 0.8;

    RoundTrip final_trip;
    final_trip.clusters = report.parsed_clusters;
    final_trip.morphemes = report.reparsed_morphemes;
    final_trip.parsed = report.reparsed;
    report.feedback = describe(report.text, final_trip,
                               report.cluster_recall, report.cluster_precision,
                               report.refinement_passes, ru);
    return report;
}

} // namespace ae::mind
