#pragma once
// ============================================================================
// AE Arch Engine (AE::mind_v1.1_Stable)
// Module 3.2: Morphemic Tokenizer Compiler (Patricia Trie + 64-bit Boolean Mask)
// Author: Andrey Firstov (IU5 BMSTU)
// ----------------------------------------------------------------------------
// Битовая раскладка 64-битной маски сочетаемости:
//   [ 0.. 7]  базовые домены смысла  (Dinitz semantic_tag 0..7, всегда активны)
//   [ 8..15]  домены, гейтируемые Dinitz-декодером по tag 8..15
//   [16..23]  интент-биты Taylor-Intent Kernel (пишет compile_morpheme_mask)
//   [24..27]  тип морфемы (PREFIX/ROOT/SUFFIX/INFLECTION)
//   [28..47]  семантический кластер корня (20 кластеров)
//   [48..63]  свободно для динамической регистрации через C ABI / FFI
// ============================================================================

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ae::mind {

struct MorphemicPatriciaTrie {
    enum class MorphemeType : std::uint8_t {
        PREFIX = 0,
        ROOT = 1,
        SUFFIX = 2,
        INFLECTION = 3
    };

    // Семантические кластеры русских корней: задают биты [28..47].
    enum class Semantic : std::uint8_t {
        GRAPH = 0,    // структура, графы, деревья, связи
        KNOW = 1,     // знание, память, смысл, онтология
        FLOW = 2,     // поток, процесс, маршрут, конвейер
        TEXT = 3,     // текст, токен, морфология, язык
        NUM = 4,      // число, математика, алгоритмы
        TIME = 5,     // время, интервал, задержка, частота
        SPACE = 6,    // пространство, геометрия, координаты
        MIND = 7,     // мозг, нейрон, когнитивистика, психика
        COMPUTE = 8,  // программы, процессоры, модули, сборка
        NET = 9,      // сети, протоколы, обмен, маршрутизация
        PHYS = 10,    // физика, энергия, волны, частицы
        CHEM = 11,    // химия, реакции, вещества
        BIO = 12,     // биология, клетки, гены, экосистемы
        MED = 13,     // медицина, здоровье, лечение
        ECON = 14,    // экономика, финансы, производство
        LAW = 15,     // право, закон, государство
        SOC = 16,     // общество, культура, образование, быт
        QUALITY = 17, // свойства, состояния, измерения, контроль
        NATURE = 18,  // природа, вещество, космос, климат
        ACTION = 19,  // действия, операции, преобразования
        MORPH = 20    // служебные морфемы: приставки, суффиксы, окончания
    };

    static constexpr std::uint64_t BASE_MASK = 0x00000000000000FFULL;

    // Идентификаторы 0..255 зарезервированы под динамическую регистрацию
    // (C ABI ae_engine_add_morpheme, Python SDK, FFI). Встроенный русский
    // инвентарь занимает диапазон начиная с FIRST_BUILTIN_ID, поэтому
    // пользовательские идентификаторы не перетирают встроенные морфемы.
    static constexpr int FIRST_BUILTIN_ID = 256;

    static constexpr std::uint64_t type_mask(MorphemeType type) noexcept {
        return 1ULL << (24 + static_cast<unsigned>(type));
    }

    static constexpr std::uint64_t cluster_mask(Semantic cluster) noexcept {
        return 1ULL << (28 + static_cast<unsigned>(cluster));
    }

    static constexpr std::uint64_t make_mask(MorphemeType type, Semantic cluster) noexcept {
        return BASE_MASK | type_mask(type) | cluster_mask(cluster);
    }

    // Результат разбора: маска сочетаемости, число найденных морфем и
    // поддержка каждого семантического кластера. Поддержка — это число
    // морфем в запросе, которые несут данный кластерный бит. Маска хранит
    // только факт наличия, поэтому два запроса с одинаковым набором битов
    // (например, "graph coloring" и "graph spectrum") раньше были
    // неразличимы; счётчики сохраняют эту информацию.
    struct MorphemeAnalysis {
        std::uint64_t mask = BASE_MASK;
        int matched_morphemes = 0;
        std::array<std::uint16_t, 20> cluster_counts{};
    };

    // Составные термины, где отдельные корни дают конкурирующие кластеры.
    // Триe матчит слова независимо, поэтому "neural network" даёт MIND + NET,
    // а "routing table" — FLOW без NET. Правило добавляет детерминированную
    // поддержку главному кластеру фразы; морфемный разбор при этом не
    // отключается, поэтому контекст сохраняется, а не подменяется.
    struct CompoundRule {
        std::string_view first;
        std::string_view second;
        std::string_view third;
        Semantic primary;
        std::uint8_t weight;
    };

    inline static constexpr CompoundRule kCompoundRules[] = {
        // English: cognition vs networking
        {"neural", "network", "", Semantic::MIND, 2},
        {"network", "of", "neurons", Semantic::MIND, 2},
        // English: graph/flow context inside networking terms
        {"network", "topology", "", Semantic::GRAPH, 2},
        {"network", "connectivity", "", Semantic::GRAPH, 2},
        {"network", "flow", "", Semantic::FLOW, 2},
        {"network", "transfer", "", Semantic::FLOW, 2},
        {"network", "throughput", "", Semantic::FLOW, 2},
        {"data", "routing", "", Semantic::FLOW, 2},
        {"stream", "routing", "", Semantic::FLOW, 2},
        {"routing", "table", "", Semantic::NET, 3},
        {"packet", "routing", "", Semantic::NET, 2},
        {"traffic", "routing", "", Semantic::NET, 2},
        {"network", "packet", "", Semantic::NET, 2},
        {"network", "traffic", "", Semantic::NET, 2},
        // Russian: те же конфликтующие пары
        {"нейронная", "сеть", "", Semantic::MIND, 2},
        {"сеть", "нейронов", "", Semantic::MIND, 2},
        {"сетевая", "топология", "", Semantic::GRAPH, 2},
        {"топология", "сети", "", Semantic::GRAPH, 2},
        {"граф", "сети", "", Semantic::GRAPH, 2},
        {"сетевой", "поток", "", Semantic::FLOW, 2},
        {"поток", "данных", "", Semantic::FLOW, 2},
        {"таблица", "маршрутов", "", Semantic::NET, 3},
        {"маршрутизация", "пакетов", "", Semantic::NET, 2},
        {"маршрутизация", "трафика", "", Semantic::NET, 2},
    };

    // ---------------------------------------------------- дерево предложения
    // Trie разбирает слова независимо. Для графового мышления нужна структура:
    // предложение режется на синтаксические чанки (по пунктуации и союзам),
    // внутри чанка выделяется голова, а поддержка кластеров агрегируется по
    // поддереву. Всё детерминировано и не использует обучение.
    struct SentenceToken {
        std::string text;
        std::size_t begin = 0;
        std::size_t end = 0;
        bool function_word = false;  // предлог/союз/детерминатор
        bool starts_chunk = true;    // с этого токена начинается новый чанк
        MorphemeAnalysis analysis{};
    };

    struct SentenceNode {
        int head_token = -1;   // индекс головного токена (для чанка/листа)
        int parent = -1;       // исходный родительский узел
        int normalized_parent = -1; // родитель после RDT-поворотов
        int depth = 0;
        int normalized_depth = 0;
        std::size_t begin_token = 0;
        std::size_t end_token = 0;
        // Поддержка кластера на этом поддереве. У чанка — агрегат всей фразы
        // (с учётом составных терминов), у листа — собственный разбор токена.
        std::array<std::uint16_t, 20> counts{};
    };

    struct TreeRotation {
        int child = -1;
        int parent = -1;
        int pivot = -1;
        int direction = 0;  // +1: child поднят, -1: child опущен
        int before_depth = 0;
        int after_depth = 0;
    };

    struct SentenceAnalysis {
        std::vector<SentenceToken> tokens;
        std::vector<SentenceNode> nodes;  // [0] — исходный корень всего предложения
        std::vector<TreeRotation> rotations;
        int normalized_root = 0;
        std::array<std::uint16_t, 20> root_counts{};
        std::array<std::uint16_t, 20> normalized_root_counts{};       // всё предложение
        std::array<std::uint16_t, 20> head_counts{};       // головы всех чанков
        std::array<std::uint16_t, 20> chunk_max_counts{};  // максимум по чанкам
        int max_depth = 0;
    };

    struct Node {
        std::string edge_label;
        int morpheme_id = -1;
        MorphemeType type = MorphemeType::ROOT;
        std::uint64_t compatibility_mask = 0;
        std::vector<std::pair<char, int>> children;
    };

    std::vector<Node> nodes;
    std::vector<std::string> dictionary;
    int next_morpheme_id = FIRST_BUILTIN_ID;

    MorphemicPatriciaTrie() {
        nodes.push_back(Node{"", -1, MorphemeType::ROOT, 0, {}});
        dictionary.resize(FIRST_BUILTIN_ID);
        seed_russian_inventory();
        seed_english_inventory();
    }

    [[nodiscard]] std::size_t morpheme_count() const noexcept {
        std::size_t count = 0;
        for (const std::string& known : dictionary) {
            if (!known.empty()) ++count;
        }
        return count;
    }

    // ---------------------------------------------------------------- trie core
    int find_child(int u, char c) const noexcept {
        for (const auto& [ch, nxt] : nodes[u].children) {
            if (ch == c) return nxt;
        }
        return -1;
    }

    bool contains(std::string_view word) const noexcept {
        for (const std::string& known : dictionary) {
            if (known == word) return true;
        }
        return false;
    }

    void bind_node(int node_index, int id, MorphemeType type, std::uint64_t mask) noexcept {
        Node& node = nodes[node_index];
        if (node.morpheme_id == -1) {
            node.morpheme_id = id;
            node.type = type;
            node.compatibility_mask = mask;
        } else {
            // Одна и та же словоформа зарегистрирована дважды: сохраняем первый
            // идентификатор и объединяем маски сочетаемости.
            node.compatibility_mask |= mask;
        }
    }

    void insert(std::string_view word, int id, MorphemeType type, std::uint64_t mask) {
        if (dictionary.size() <= static_cast<std::size_t>(id)) {
            dictionary.resize(id + 1);
        }
        dictionary[id] = std::string(word);

        int cur = 0;
        std::size_t pos = 0;
        while (pos < word.size()) {
            char c = word[pos];
            int nxt = find_child(cur, c);
            if (nxt == -1) {
                int new_idx = static_cast<int>(nodes.size());
                nodes.push_back(Node{std::string(word.substr(pos)), id, type, mask, {}});
                nodes[cur].children.push_back({c, new_idx});
                return;
            }
            std::string_view label = nodes[nxt].edge_label;
            std::size_t common = 0;
            while (common < label.size() && pos + common < word.size() &&
                   label[common] == word[pos + common]) {
                ++common;
            }
            if (common == label.size()) {
                cur = nxt;
                pos += common;
            } else {
                int split_idx = static_cast<int>(nodes.size());
                Node split_node{std::string(label.substr(0, common)), -1, MorphemeType::ROOT, 0, {}};
                nodes[nxt].edge_label = std::string(label.substr(common));
                split_node.children.push_back({nodes[nxt].edge_label[0], nxt});

                for (auto& [ch, child_idx] : nodes[cur].children) {
                    if (child_idx == nxt) {
                        child_idx = split_idx;
                        break;
                    }
                }
                nodes.push_back(std::move(split_node));

                pos += common;
                if (pos == word.size()) {
                    bind_node(split_idx, id, type, mask);
                } else {
                    int leaf_idx = static_cast<int>(nodes.size());
                    nodes.push_back(Node{std::string(word.substr(pos)), id, type, mask, {}});
                    nodes[split_idx].children.push_back({word[pos], leaf_idx});
                }
                return;
            }
        }
        bind_node(cur, id, type, mask);
    }

    // ------------------------------------------------------------ seeding tables
    void seed_roots(Semantic cluster,
                    std::initializer_list<std::string_view> forms) {
        for (std::string_view form : forms) {
            if (contains(form)) continue;
            insert(form, next_morpheme_id++, MorphemeType::ROOT,
                   make_mask(MorphemeType::ROOT, cluster));
        }
    }

    void seed_affixes(MorphemeType type,
                      std::initializer_list<std::string_view> forms) {
        for (std::string_view form : forms) {
            if (contains(form)) continue;
            insert(form, next_morpheme_id++, type, make_mask(type, Semantic::MORPH));
        }
    }

    // --------------------------------------------------------- русский инвентарь
    void seed_russian_inventory() {
        // -- приставки ---------------------------------------------------------
        seed_affixes(MorphemeType::PREFIX, {
            // исконно русские
            "пере", "под", "подо", "над", "пред", "предо", "при", "про", "пре",
            "пра", "раз", "разо", "рас", "роз", "рос", "без", "бес", "воз",
            "возо", "вос", "вз", "взо", "вс", "низ", "нис", "из", "изо", "ис",
            "от", "ото", "об", "обо", "обез", "обес", "до", "за", "на", "наи",
            "не", "недо", "по", "поза", "со", "вы",
            // продуктивные иноязычные
            "противо", "электро", "магнито", "псевдо", "квази", "крипто",
            "сейсмо", "стерео", "стресс", "социо", "энерго", "нейро", "гипер",
            "гипо", "инфра", "интра", "интер", "контр", "макро", "микро",
            "мульти", "супер", "сверх", "ультра", "экстра", "анти", "архи",
            "транс", "ретро", "прото", "пост", "моно", "поли", "нано", "мега",
            "мета", "пан", "экс", "дез", "дис", "орто", "эндо", "экзо", "обще",
            "полу", "разно", "равно", "много", "мало", "низко", "высоко",
            "тонко", "широко", "крупно", "мелко", "быстро", "медленно",
            "светло", "темно", "около", "после", "через", "чрез", "вне",
            "внутри", "между", "меж", "суб", "нео", "авиа", "авто",
            "агро", "астро", "аудио", "аэро", "био", "вело", "видео", "вице",
            "гео", "гелио", "гидро", "зоо", "изо", "кибер", "кино", "лже",
            "медиа", "метео", "мото", "нарко", "радио", "само", "спец", "теле",
            "термо", "топо", "турбо", "фото", "фоно", "хроно", "эко", "этно"
        });

        // -- корни: структура и графы ----------------------------------------
        seed_roots(Semantic::GRAPH, {
            "граф", "дерев", "корен", "корнев", "вершин", "ребр", "узл", "связ",
            "структур", "иерарх", "кластер", "компонент", "композиц",
            "декомпозиц", "подграф", "гиперграф", "изоморф", "топологи",
            "контур", "обход", "рекурси", "ветв", "предок", "потомок", "каркас",
            "остов", "шарнир", "смежн", "инцидент", "валентн", "степен",
            "бинарн", "сбалансир", "разрежен", "плотн", "взвешен", "ориентир",
            "ациклическ", "эйлеров", "гамильтонов"
        });

        // -- корни: знание, память, смысл -------------------------------------
        seed_roots(Semantic::KNOW, {
            "знан", "памят", "смысл", "мысл", "понят", "разум", "интеллект",
            "сознани", "опыт", "инфо", "данн", "факт", "истин", "гипотез",
            "концепт", "категори", "онтологи", "тезаурус", "определ", "суждени",
            "логик", "инференц", "аксиом", "парадигм", "дискурс", "нарратив",
            "контекст", "релевантн", "достоверн", "заблужд", "эрудиц", "эксперт",
            "компетенц", "обучен", "запомина", "воспроизвед", "ассоциац",
            "семантик"
        });

        // -- корни: поток, процесс, маршрут -----------------------------------
        seed_roots(Semantic::FLOW, {
            "поток", "течен", "движен", "процесс", "операц", "действ", "цикл",
            "итерац", "переход", "каскад", "маршрут", "траектор", "канал",
            "пропускн", "конвейер", "очередь", "буфер", "пайплайн", "этап",
            "фаз", "шаг", "ритм", "темп", "ускорен", "замедлен", "останов",
            "завершен", "повтор", "последов", "параллел", "синхрон", "асинхрон",
            "конкурен", "планировщик", "диспетчер", "распредел", "балансиров",
            "туннел", "затор"
        });

        // -- корни: текст, язык, токены ----------------------------------------
        seed_roots(Semantic::TEXT, {
            "текст", "токен", "слов", "морфем", "лексем", "корпус", "фраз",
            "предложен", "абзац", "документ", "символ", "букв", "знак", "язык",
            "реч", "граммат", "синтакс", "алфавит", "шрифт", "вербал", "глосс",
            "перевод", "транслит", "транскрипц", "пунктуац", "орфографи",
            "стилистик", "риторик", "метафор", "идиом", "жаргон", "диалект",
            "падеж", "склонен", "спряжен", "местоимен", "прилагательн",
            "существительн", "глагол", "причасти", "деепричасти", "союз",
            "предлог", "частиц", "междомети"
        });

        // -- корни: число, математика, алгоритмы -------------------------------
        seed_roots(Semantic::NUM, {
            "числ", "числен", "вектор", "матриц", "тензор", "функци", "полином",
            "тейлор", "градиент", "оптимизац", "максим", "миним", "сумм",
            "интеграл", "дифференциал", "вероятн", "статистик", "индекс",
            "коэффициен", "формул", "теорем", "множеств", "групп", "алгоритм",
            "вычисл", "метрик", "скаляр", "спектр", "сингулярн", "аппроксим",
            "интерполяц", "экстраполяц", "сходим", "предел", "производн",
            "логарифм", "экспонент", "синус", "косинус", "факториал", "дискретн",
            "непрерывн", "целочислен", "остаток", "делен", "умножен", "сложен",
            "вычитан", "дробн"
        });

        // -- корни: время ------------------------------------------------------
        seed_roots(Semantic::TIME, {
            "врем", "времен", "секунд", "минут", "час", "период", "интервал",
            "задержк", "латентн", "длительн", "частот", "момент", "эпох",
            "прогрес", "истори", "календар", "сутк", "недел", "месяц", "годов",
            "век", "современ", "древн", "будущ", "прошл", "настоящ", "таймер",
            "таймаут", "дедлайн", "расписан", "хронологи", "мгновенн",
            "долговремен", "кратковремен"
        });

        // -- корни: пространство, геометрия ------------------------------------
        seed_roots(Semantic::SPACE, {
            "пространств", "координ", "позиц", "мест", "обла", "точк", "лини",
            "плоскост", "объем", "размер", "геометри", "расстоя", "ближн",
            "дальн", "границ", "направлен", "ось", "широт", "долгот", "высот",
            "глубин", "поверхност", "крив", "окружност", "сфер", "куб", "угол",
            "перпендикуляр", "параллельн", "сегмент", "отрезок", "полигон",
            "решетк", "масштаб", "симметри", "проекц", "перспектив"
        });

        // -- корни: мозг, когнитивистика, психика ------------------------------
        seed_roots(Semantic::MIND, {
            "мозг", "нейрон", "синапс", "когнит", "восприят", "вниман", "эмоц",
            "мотив", "решени", "выбор", "оценк", "прогноз", "предсказан",
            "обобщен", "абстракц", "аналоги", "интуиц", "вол", "жела", "характер",
            "темперамент", "личност", "психолог", "поведен", "рефлекс",
            "инстинкт", "осознан", "подсознан", "медитац", "стресс", "тревог",
            "радост", "страдан", "страст", "довери", "сомнен", "уверенн",
            "критическ", "креативн", "инсайт"
        });

        // -- корни: вычислительная техника -------------------------------------
        seed_roots(Semantic::COMPUTE, {
            "програм", "код", "компил", "машин", "процессор", "кэш", "регистр",
            "систем", "интерфейс", "модул", "библиотек", "архитектур",
            "платформ", "сервер", "клиент", "байт", "бит", "файл", "диск", "ядр",
            "драйвер", "отладк", "тестирован", "сборк", "линковк", "ассемблер",
            "интерпретатор", "транслятор", "декомпил", "байткод", "опкод",
            "инструкц", "команд", "исполняем", "загрузчик", "демон",
            "конфигурац", "зависимост", "версион", "релиз", "патч", "обновлен",
            "ключ", "шифр"
        });

        // -- корни: сети и связь -----------------------------------------------
        seed_roots(Semantic::NET, {
            "сет", "интернет", "протокол", "пакет", "запрос", "ответ", "передач",
            "соединен", "порт", "адрес", "домен", "маршрутизац", "коммутац",
            "трафик", "шлюз", "прокси", "брандмауэр", "файрвол", "авторизац",
            "аутентификац", "сесси", "куки", "репликац", "сокет", "модем",
            "роутер", "свитч", "спутник", "телекоммуникац", "радиоволн"
        });

        // -- корни: физика ------------------------------------------------------
        seed_roots(Semantic::PHYS, {
            "физик", "энерг", "тепл", "свет", "звук", "волн", "частиц", "атом",
            "молекул", "кварк", "электр", "магнит", "гравитац", "масс", "импульс",
            "давлен", "температур", "плотност", "вязкост", "колеба", "резонанс",
            "дифракц", "интерференц", "поляризац", "фотон", "электрон", "протон",
            "нейтрон", "плазм", "вакуум", "энтропи", "энтальпи",
            "термодинамик"
        });

        // -- корни: химия -------------------------------------------------------
        seed_roots(Semantic::CHEM, {
            "хими", "реакц", "веществ", "кислот", "щелоч", "раствор", "катализ",
            "полимер", "оксид", "соль", "ион", "молярн", "концентрац", "осадок",
            "испарен", "плавлен", "кипен", "горен", "окислен", "восстановлен",
            "синтез", "элемент", "изотоп", "радиоактивн", "реагент",
            "растворител", "кристаллизац", "электролиз"
        });

        // -- корни: биология ----------------------------------------------------
        seed_roots(Semantic::BIO, {
            "биолог", "клетк", "ген", "днк", "рнк", "геном", "белок", "организм",
            "эволюц", "растен", "животн", "эколог", "популяц", "экосистем", "вид",
            "род", "фотосинтез", "метаболизм", "фермент", "гормон", "бактери",
            "микроб", "гриб", "водоросл", "насеком", "птиц", "рыб", "млекопита",
            "биосфер", "симбиоз", "мутац", "наследствен", "селекц"
        });

        // -- корни: медицина ----------------------------------------------------
        seed_roots(Semantic::MED, {
            "медиц", "здоров", "болезн", "лекарств", "диагноз", "лечен",
            "пациент", "клиник", "иммун", "вирус", "инфекц", "симптом", "синдром",
            "терапи", "хирург", "реабилитац", "вакцин", "антител", "аллерг",
            "опухол", "травм", "анестези", "рецепт", "дозировк", "профилактик",
            "эпидеми", "пандеми", "кров", "сердц"
        });

        // -- корни: экономика и производство ------------------------------------
        seed_roots(Semantic::ECON, {
            "эконом", "финанс", "деньг", "стоим", "цен", "рынок", "торгов",
            "производств", "предприят", "бизнес", "банк", "кредит", "налог",
            "бюджет", "инвестиц", "капитал", "прибыл", "убыт", "расход", "доход",
            "инфляц", "дефицит", "спрос", "конкуренц", "монопол", "акци",
            "облигац", "валюта", "бирж", "аудит", "бухгалтер", "логистик",
            "поставк", "склад", "сбыт", "маркетинг", "реклам", "бренд"
        });

        // -- корни: право и государство -----------------------------------------
        seed_roots(Semantic::LAW, {
            "прав", "закон", "суд", "государств", "конституц", "договор",
            "ответств", "юридическ", "лиценз", "патент", "авторств", "норм",
            "кодекс", "устав", "регламент", "юрисдикц", "прокурор", "адвокат",
            "нотариус", "приговор", "преступ", "наказан", "амнисти", "таможн",
            "сертификац", "стандартизац"
        });

        // -- корни: общество, культура, образование ------------------------------
        seed_roots(Semantic::SOC, {
            "обществ", "культур", "искусств", "музык", "живопис", "литератур",
            "спортив", "игр", "семь", "человек", "люд", "народ", "город",
            "стран", "транспор", "дорог", "путешеств", "традиц", "обыча",
            "религи", "философ", "этик", "морал", "образован", "школ",
            "университет", "студент", "преподават", "учител", "урок", "лекц",
            "семинар", "экзамен", "диплом", "наук"
        });

        // -- корни: свойства, состояния, измерения -------------------------------
        seed_roots(Semantic::QUALITY, {
            "качеств", "свойств", "признак", "состоян", "изменен", "развити",
            "рост", "уменьшен", "увеличен", "сравнен", "различи", "сходств",
            "порядок", "хаос", "риск", "надежн", "точност", "ошибк", "проверк",
            "контрол", "измерен", "тест", "валидац", "верификац", "калибровк",
            "эталон", "стандарт", "критери", "показател", "параметр", "характ",
            "спецификац", "требовани", "ограничен", "допуск", "погрешност",
            "стабильн", "устойчив", "масштабир"
        });

        // -- корни: природа, вещество, космос ------------------------------------
        seed_roots(Semantic::NATURE, {
            "земл", "вод", "воздух", "огон", "лес", "трав", "цвет", "камен",
            "стекл", "метал", "пластик", "нефт", "газ", "космос", "планет",
            "звезд", "луна", "солнц", "океан", "море", "река", "озер", "горн",
            "поле", "пустын", "климат", "погод", "атмосфер", "минерал", "руда",
            "уголь", "алмаз", "кристалл", "песок", "глин", "почв"
        });

        // -- корни: действия и операции -----------------------------------------
        seed_roots(Semantic::ACTION, {
            "создан", "построен", "удален", "добавлен", "поиск", "найд",
            "хранен", "чтени", "запис", "копир", "слиян", "сортировк",
            "фильтрац", "агрегац", "генерац", "трансформац", "конвертац",
            "кодирован", "декодирован", "сжати", "распаковк", "загрузк",
            "выгрузк", "импорт", "экспорт", "миграц", "резервн", "очистк",
            "вставк", "извлечен", "анализ", "классификац", "кластеризац",
            "регресси", "мониторинг", "профилирован", "логирован", "трассировк",
            "рефактор", "инкремент", "декремент", "инициализац", "аллокац",
            "блокировк", "мьютекс", "семафор"
        });

        // -- суффиксы -----------------------------------------------------------
        seed_affixes(MorphemeType::SUFFIX, {
            "ированность", "ируемость", "абельность", "ационность",
            "тельность", "ительность", "енность", "ирование", "ование",
            "изация", "фикация", "ничество", "ичество", "тельство",
            "ительство", "оведение", "ведение", "логия", "графия", "метрия",
            "номия", "скопия", "ация", "зация", "ение", "ание", "изм", "ист",
            "ическ", "альн", "ельн", "онн", "енн", "ческ", "ичн", "ск", "лив",
            "чив", "оват", "еват", "еньк", "оньк", "ушк", "юшк", "ышк", "ишк",
            "ищ", "инк", "ость", "есть", "ность", "мость", "имость", "тель",
            "чик", "щик", "ник", "ниц", "ница", "ств", "еств", "овств", "арь",
            "ырь", "атор", "ёр", "ок", "ек", "ик", "ец", "ак", "як", "ач", "ич",
            "ович", "евич"
        });

        // -- окончания и глагольные форманты ------------------------------------
        seed_affixes(MorphemeType::INFLECTION, {
            "ться", "тся", "ого", "его", "ому", "ему", "ыми", "ими", "ых", "их",
            "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой", "ей", "ую",
            "юю", "ами", "ями", "ах", "ях", "ам", "ям", "ию", "ия", "ии", "ью",
            "ем", "ом", "ов", "ев", "ин", "шь", "те", "ет", "ют", "ут", "ат",
            "ят", "ит", "им", "ете", "ите", "ешь", "ишь", "ует", "ирует", "ал",
            "ил", "ел", "ла", "ло", "ли", "ть", "ти", "чь", "а", "я", "о", "е",
            "ы", "и", "у", "ю"
        });
    }

    // --------------------------------------------------------- английский инвентарь
    void seed_english_inventory() {
        // -- приставки ---------------------------------------------------------
        seed_affixes(MorphemeType::PREFIX, {
            "counter", "electro", "thermo", "techno", "hydro", "photo", "micro",
            "macro", "hyper", "hypo", "inter", "intra", "infra", "extra",
            "ultra", "supra", "super", "under", "over", "trans", "proto",
            "pseudo", "quasi", "multi", "poly", "mono", "semi", "anti", "auto",
            "bio", "geo", "cyber", "neuro", "info", "meta", "para", "peri",
            "post", "pre", "pro", "retro", "tele", "mega", "giga", "kilo",
            "milli", "centi", "nano", "mini", "mid", "mis", "non", "out",
            "self", "well", "fore", "with", "dis", "en", "ex", "co", "de",
            "il", "im", "in", "ir", "re", "un", "up", "sub", "sur", "mal",
            "tri"
        });

        // -- корни: graph / structure -----------------------------------------
        seed_roots(Semantic::GRAPH, {
            "graph", "tree", "node", "edge", "vertex", "vertic", "link",
            "root", "leaf", "branch", "cluster", "component",
            "structure", "hierarchy", "topology", "cycle", "loop", "mesh",
            "grid", "chain", "spanning", "adjacency", "degree", "connectivity",
            "forest", "subgraph", "traverse"
        });

        // -- корни: knowledge / memory ----------------------------------------
        seed_roots(Semantic::KNOW, {
            "know", "memory", "memor", "meaning", "mind", "concept", "category",
            "ontology", "fact", "truth", "belief", "knowledge", "context",
            "semantic", "inference", "logic", "reason", "axiom", "theory",
            "data", "information", "inform", "insight", "wisdom", "learn",
            "remember", "forget", "recall", "understand", "explain", "thought",
            "idea"
        });

        // -- корни: flow / process --------------------------------------------
        seed_roots(Semantic::FLOW, {
            "flow", "stream", "process", "pipeline", "route", "routing",
            "channel", "queue", "buffer", "stage", "phase", "transfer",
            "propagate", "cascade", "conveyor", "scheduler", "dispatch",
            "distributed", "parallel", "sync", "async", "concur", "throughput",
            "bandwidth", "latency", "rate", "through"
        });

        // -- корни: text / language -------------------------------------------
        seed_roots(Semantic::TEXT, {
            "text", "token", "word", "morph", "lexeme", "corpus", "phrase",
            "sentence", "paragraph", "document", "symbol", "letter", "grammar",
            "syntax", "alphabet", "font", "verbal", "gloss", "translat",
            "transcript", "punct", "style", "rhetoric", "metaphor", "idiom",
            "dialect", "noun", "verb", "adjective", "adverb", "pronoun",
            "preposition", "conjunction", "tense"
        });

        // -- корни: numbers / math --------------------------------------------
        seed_roots(Semantic::NUM, {
            "number", "numeric", "vector", "matrix", "tensor", "function",
            "polynomial", "taylor", "gradient", "optim", "maxim", "minim",
            "sum", "integral", "differenti", "probability", "statistic",
            "index", "coefficient", "formula", "theorem", "algorithm",
            "compute", "metric", "scalar", "spectrum", "singular", "approxim",
            "interpola", "extrapola", "converge", "limit", "derivative",
            "logarithm", "exponent", "sine", "cosine", "factorial", "discrete",
            "continuous", "integer", "fraction", "remainder", "divide",
            "multiply", "subtract"
        });

        // -- корни: time -------------------------------------------------------
        seed_roots(Semantic::TIME, {
            "time", "second", "minute", "hour", "period", "interval", "delay",
            "duration", "frequency", "moment", "epoch", "progress", "history",
            "calendar", "week", "month", "year", "century", "modern", "ancient",
            "future", "past", "present", "timer", "timeout", "deadline",
            "schedule", "chrono", "instant"
        });

        // -- корни: space / geometry -------------------------------------------
        seed_roots(Semantic::SPACE, {
            "space", "coordinate", "position", "location", "area", "point",
            "line", "plane", "volume", "size", "geometry", "distance", "border",
            "direction", "axis", "latitude", "longitude", "height", "depth",
            "surface", "curve", "circle", "sphere", "cube", "angle",
            "perpendicular", "segment", "polygon", "scale", "symmetric",
            "project", "perspective"
        });

        // -- корни: brain / cognition ------------------------------------------
        seed_roots(Semantic::MIND, {
            "brain", "neuron", "synapse", "cognit", "perception", "attention",
            "emotion", "motive", "decision", "choice", "evaluation", "forecast",
            "predict", "abstract", "analog", "intuition", "character",
            "temperament", "personality", "psycholog", "behavior", "reflex",
            "instinct", "conscious", "subconscious", "meditation", "stress",
            "anxiety", "passion", "trust", "doubt", "confidence", "critical",
            "creative", "insight"
        });

        // -- корни: computing ---------------------------------------------------
        seed_roots(Semantic::COMPUTE, {
            "program", "code", "compile", "machine", "processor", "cache",
            "register", "system", "interface", "module", "library",
            "architecture", "platform", "server", "client", "byte", "file",
            "disk", "kernel", "driver", "debug", "test", "build", "assembler",
            "interpreter", "translator", "bytecode", "opcode", "instruction",
            "command", "execute", "loader", "daemon", "config", "dependency",
            "version", "release", "patch", "update", "cipher"
        });

        // -- корни: networking ---------------------------------------------------
        seed_roots(Semantic::NET, {
            "internet", "network", "protocol", "packet", "request", "response",
            "transmit", "connect", "port", "address", "domain", "switch",
            "traffic", "gateway", "proxy", "firewall", "auth", "session",
            "cookie", "replica", "socket", "modem", "router", "satellite",
            "telecom", "wireless"
        });

        // -- корни: physics ------------------------------------------------------
        seed_roots(Semantic::PHYS, {
            "physic", "energy", "heat", "light", "sound", "wave", "particle",
            "atom", "molecule", "quark", "electric", "magnet", "gravity",
            "mass", "momentum", "pressure", "temperature", "density",
            "viscosity", "oscillat", "resonance", "diffraction",
            "interference", "polarization", "photon", "electron", "proton",
            "neutron", "plasma", "vacuum", "entropy", "thermodynamic"
        });

        // -- корни: chemistry -----------------------------------------------------
        seed_roots(Semantic::CHEM, {
            "chem", "reaction", "substance", "acid", "alkali", "solution",
            "catalysis", "polymer", "oxide", "salt", "molar", "concentration",
            "sediment", "evaporat", "melting", "boiling", "burning",
            "oxidation", "reduction", "synthesis", "element", "isotope",
            "radioactiv", "reagent", "solvent", "crystallization",
            "electrolysis"
        });

        // -- корни: biology --------------------------------------------------------
        seed_roots(Semantic::BIO, {
            "biolog", "cell", "gene", "genome", "protein", "organism",
            "evolution", "plant", "animal", "ecolog", "population", "ecosystem",
            "species", "photosynthesis", "metabolism", "enzyme", "hormone",
            "bacteria", "microbe", "fungus", "algae", "insect", "bird", "fish",
            "mammal", "biosphere", "symbiosis", "mutation", "heredity",
            "selection"
        });

        // -- корни: medicine --------------------------------------------------------
        seed_roots(Semantic::MED, {
            "medic", "health", "disease", "drug", "diagnosis", "treatment",
            "patient", "clinic", "immune", "virus", "infection", "symptom",
            "syndrome", "therapy", "surgery", "rehabilitation", "vaccine",
            "antibody", "allergy", "tumor", "trauma", "anesthesia",
            "prescription", "dosage", "prevention", "epidemic", "pandemic",
            "blood", "heart"
        });

        // -- корни: economics ---------------------------------------------------------
        seed_roots(Semantic::ECON, {
            "econom", "finance", "money", "cost", "price", "market", "trade",
            "production", "enterprise", "business", "bank", "credit", "tax",
            "budget", "invest", "capital", "profit", "loss", "expense",
            "income", "inflation", "deficit", "demand", "competition",
            "monopoly", "stock", "bond", "currency", "exchange", "audit",
            "accounting", "logistic", "supply", "warehouse", "sales",
            "marketing", "advertising", "brand"
        });

        // -- корни: law ---------------------------------------------------------------
        seed_roots(Semantic::LAW, {
            "law", "right", "court", "state", "constitution", "contract",
            "responsibility", "legal", "license", "patent", "copyright",
            "norm", "charter", "regulation", "jurisdiction", "prosecutor",
            "lawyer", "notary", "verdict", "crime", "punishment", "amnesty",
            "customs", "certification", "standardization"
        });

        // -- корни: society / culture --------------------------------------------------
        seed_roots(Semantic::SOC, {
            "society", "culture", "art", "music", "paint", "literature",
            "sport", "game", "family", "human", "people", "nation", "city",
            "country", "transport", "road", "travel", "tradition", "custom",
            "religion", "philosophy", "ethic", "moral", "education", "school",
            "university", "student", "teacher", "lesson", "lecture", "seminar",
            "exam", "diploma", "science"
        });

        // -- корни: quality / state / measurement ----------------------------------------
        seed_roots(Semantic::QUALITY, {
            "quality", "property", "feature", "change", "development", "growth",
            "decrease", "increase", "compare", "difference", "similarity",
            "order", "chaos", "risk", "reliability", "accuracy", "error",
            "check", "control", "measure", "validate", "verify", "calibrate",
            "standard", "criteria", "parameter", "specification",
            "requirement", "constraint", "tolerance", "stability", "robust"
        });

        // -- корни: nature ---------------------------------------------------------------
        seed_roots(Semantic::NATURE, {
            "earth", "water", "air", "fire", "grass", "flower", "stone",
            "glass", "metal", "plastic", "oil", "gas", "cosmos", "planet",
            "star", "moon", "sun", "ocean", "sea", "river", "lake", "mountain",
            "field", "desert", "climate", "weather", "atmosphere", "mineral",
            "ore", "coal", "diamond", "crystal", "sand", "clay", "soil"
        });

        // -- корни: actions ---------------------------------------------------------------
        seed_roots(Semantic::ACTION, {
            "create", "build", "delete", "add", "search", "find", "store",
            "read", "write", "copy", "merge", "sort", "filter", "aggregate",
            "generate", "transform", "convert", "encode", "decode", "compress",
            "decompress", "encrypt", "decrypt", "load", "import", "export",
            "migrate", "backup", "restore", "clean", "insert", "extract",
            "analyze", "classify", "regress", "monitor", "profile", "trace",
            "refactor", "increment", "decrement", "lock", "mutex", "semaphore"
        });

        // -- суффиксы -----------------------------------------------------------------------
        seed_affixes(MorphemeType::SUFFIX, {
            "ization", "isation", "ational", "ability", "ibility", "ation",
            "ition", "ology", "ography", "ometry", "ically", "ative", "ivity",
            "ness", "ment", "tion", "sion", "ance", "ence", "ship", "hood",
            "ward", "wise", "less", "ful", "able", "ible", "ism", "ist",
            "ity", "ify", "ize", "ise", "ous", "ive", "ary", "ory", "al",
            "ic", "en", "ish", "ant", "ent", "age", "dom"
        });

        // -- окончания ------------------------------------------------------------------------
        seed_affixes(MorphemeType::INFLECTION, {
            "ing", "est", "ed", "es", "er", "ly", "s"
        });
    }

    // ------------------------------------------------------------------- lookup
    static constexpr char ascii_lower(char c) noexcept {
        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    }

    static constexpr bool label_matches(std::string_view text, std::size_t pos,
                                       std::string_view label) noexcept {
        for (std::size_t i = 0; i < label.size(); ++i) {
            if (ascii_lower(text[pos + i]) != label[i]) return false;
        }
        return true;
    }

    static constexpr bool is_word_byte(unsigned char byte) noexcept {
        return byte >= 0x80 || (byte >= '0' && byte <= '9') ||
               (byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
               byte == '_' || byte == '\'';
    }

    // Пословный жадный longest-match: внутри слова берётся самая длинная
    // зарегистрированная морфема, затем разбор продолжается с её конца.
    // Такой проход не считает вложенные совпадения ("аст" внутри "астро")
    // и не завышает matched_morphemes на словаре в 1000+ единиц.
    static bool token_equals(std::string_view text, std::size_t begin,
                             std::size_t end, std::string_view token) noexcept {
        if (end - begin != token.size()) return false;
        for (std::size_t i = 0; i < token.size(); ++i) {
            if (ascii_lower(text[begin + i]) != ascii_lower(token[i])) {
                return false;
            }
        }
        return true;
    }

    static bool next_token(std::string_view text, std::size_t from,
                           std::size_t& begin, std::size_t& end) noexcept {
        std::size_t cursor = from;
        while (cursor < text.size() &&
               !is_word_byte(static_cast<unsigned char>(text[cursor]))) {
            ++cursor;
        }
        if (cursor == text.size()) return false;
        begin = cursor;
        while (cursor < text.size() &&
               is_word_byte(static_cast<unsigned char>(text[cursor]))) {
            ++cursor;
        }
        end = cursor;
        return true;
    }

    // Служебные слова не становятся головой чанка, но остаются в дереве как
    // зависимости. Списки намеренно короткие и только высокочастотные, чтобы
    // не «съесть» содержательные слова.
    inline static constexpr std::string_view kPrepositions[] = {
        "в", "во", "на", "по", "с", "со", "из", "от", "до", "для", "при",
        "о", "об", "обо", "про", "через", "между", "над", "под", "без",
        "к", "ко", "у", "за", "около", "после", "перед", "вокруг", "сквозь",
        "среди", "кроме", "ради", "благодаря", "согласно", "вопреки",
        "вместо", "вдоль", "поперёк",
        "in", "on", "at", "by", "for", "from", "to", "with", "without",
        "about", "through", "between", "over", "under", "into", "via",
        "across", "along", "around", "against", "during", "before", "after",
        "above", "below", "per", "within", "upon", "onto", "toward",
        "towards", "off", "out", "up", "down",
    };

    inline static constexpr std::string_view kConjunctions[] = {
        "и", "а", "но", "или", "либо",
        "and", "or", "but", "nor",
    };

    inline static constexpr std::string_view kDeterminers[] = {
        "не", "ни", "то", "как", "что", "это", "же", "бы", "ли", "так",
        "тоже", "также", "эти", "этот", "эта", "тот", "та", "те", "все",
        "весь", "вся", "каждый", "любой", "некоторый",
        "the", "a", "an", "not", "is", "are", "was", "were", "be", "been",
        "being", "it", "its", "this", "that", "these", "those", "we", "you",
        "they", "he", "she", "his", "her", "their", "our", "my", "your",
        "then", "also", "as", "than",
    };

    template <std::size_t N>
    static bool token_in_list(std::string_view text, std::size_t begin,
                              std::size_t end,
                              const std::string_view (&list)[N]) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            if (token_equals(text, begin, end, list[i])) return true;
        }
        return false;
    }

    static bool is_conjunction(std::string_view text, std::size_t begin,
                               std::size_t end) noexcept {
        return token_in_list(text, begin, end, kConjunctions);
    }

    static bool is_function_word(std::string_view text, std::size_t begin,
                                 std::size_t end) noexcept {
        return is_conjunction(text, begin, end) ||
               token_in_list(text, begin, end, kPrepositions) ||
               token_in_list(text, begin, end, kDeterminers);
    }

    static const CompoundRule* match_compound(std::string_view text,
                                               std::size_t begin,
                                               std::size_t end) noexcept {
        for (const CompoundRule& rule : kCompoundRules) {
            if (!token_equals(text, begin, end, rule.first)) continue;
            std::size_t second_begin = 0;
            std::size_t second_end = 0;
            if (!next_token(text, end, second_begin, second_end)) continue;
            if (!token_equals(text, second_begin, second_end, rule.second)) continue;
            if (!rule.third.empty()) {
                std::size_t third_begin = 0;
                std::size_t third_end = 0;
                if (!next_token(text, second_end, third_begin, third_end)) continue;
                if (!token_equals(text, third_begin, third_end, rule.third)) continue;
            }
            return &rule;
        }
        return nullptr;
    }

    __attribute__((noinline)) MorphemeAnalysis compile_morpheme_analysis(
        std::string_view text, const std::array<float, 16>& intent) const noexcept {
        MorphemeAnalysis analysis;
        std::uint64_t& mask = analysis.mask;

        std::size_t cursor = 0;
        while (cursor < text.size()) {
            while (cursor < text.size() &&
                   !is_word_byte(static_cast<unsigned char>(text[cursor]))) {
                ++cursor;
            }
            if (cursor == text.size()) break;

            const std::size_t word_end = [&] {
                std::size_t end = cursor;
                while (end < text.size() &&
                       is_word_byte(static_cast<unsigned char>(text[end]))) {
                    ++end;
                }
                return end;
            }();

            // Составной термин добавляет поддержку главному кластеру до
            // морфемного разбора. Например, "routing table" получает NET,
            // хотя корень routing сам по себе относится к FLOW.
            if (const CompoundRule* compound = match_compound(text, cursor, word_end)) {
                mask |= cluster_mask(compound->primary);
                analysis.cluster_counts[static_cast<std::size_t>(compound->primary)] +=
                    compound->weight;
            }

            std::size_t pos = cursor;
            while (pos < word_end) {
                int cur = 0;
                std::size_t scan = pos;
                int best_node = -1;
                std::size_t best_end = pos;
                while (scan < word_end) {
                    int nxt = find_child(cur, ascii_lower(text[scan]));
                    if (nxt == -1) break;
                    std::string_view label = nodes[nxt].edge_label;
                    if (word_end - scan < label.size() ||
                        !label_matches(text, scan, label)) {
                        break;
                    }
                    scan += label.size();
                    cur = nxt;
                    if (nodes[cur].morpheme_id != -1) {
                        best_node = cur;
                        best_end = scan;
                    }
                }
                if (best_node == -1) {
                    // Разбор слова непрерывен: неизвестный участок обрывает
                    // цепочку, иначе поиск с середины слова даёт ложные
                    // срабатывания ("кот" -> приставка "от").
                    break;
                }
                const std::uint64_t compatibility = nodes[best_node].compatibility_mask;
                mask |= compatibility;
                for (int cluster = 0; cluster < 20; ++cluster) {
                    if (compatibility & cluster_mask(static_cast<Semantic>(cluster))) {
                        ++analysis.cluster_counts[static_cast<std::size_t>(cluster)];
                    }
                }
                ++analysis.matched_morphemes;
                pos = best_end;
            }
            cursor = word_end;
        }

        for (std::size_t b = 0; b < 16; ++b) {
            if (intent[b] > -0.25f) {
                mask |= (1ULL << (b + 8));
            }
        }
        return analysis;
    }

    // Строит дерево предложения: чанки по пунктуации/союзам, в каждом чанке
    // голова (последнее содержательное слово) и зависимые служебные слова.
    // Поддержка кластеров считается по поддереву чанка, поэтому локальная
    // фраза ("routing table", "network topology") не растворяется в общем
    // мешке слов. Голова получает поддержку составного термина, заканчиваю-
    // щегося на ней, — это обобщает прежние жёсткие правила на структуру.
    __attribute__((noinline)) SentenceAnalysis compile_sentence_analysis(
        std::string_view text, const std::array<float, 16>& intent) const noexcept {
        SentenceAnalysis out;

        std::size_t cursor = 0;
        bool starts_chunk = true;
        while (cursor < text.size()) {
            bool punctuation_break = false;
            while (cursor < text.size() &&
                   !is_word_byte(static_cast<unsigned char>(text[cursor]))) {
                const char ch = text[cursor];
                if (ch == ',' || ch == ';' || ch == ':' || ch == '!' ||
                    ch == '?' || ch == '(' || ch == ')' || ch == '[' ||
                    ch == ']' || ch == '{' || ch == '}' || ch == '.' ||
                    ch == '\n') {
                    punctuation_break = true;
                }
                ++cursor;
            }
            if (cursor == text.size()) break;

            const std::size_t begin = cursor;
            while (cursor < text.size() &&
                   is_word_byte(static_cast<unsigned char>(text[cursor]))) {
                ++cursor;
            }
            const std::size_t end = cursor;

            SentenceToken token;
            token.text = std::string(text.substr(begin, end - begin));
            token.begin = begin;
            token.end = end;
            token.function_word = is_function_word(text, begin, end);
            token.starts_chunk = starts_chunk || punctuation_break;
            // Разбор одного токена: составные термины требуют двух слов, они
            // добавятся ниже на уровне чанка.
            token.analysis = compile_morpheme_analysis(token.text, intent);
            out.tokens.push_back(token);

            starts_chunk = is_conjunction(text, begin, end);
        }

        if (out.tokens.empty()) return out;

        SentenceNode root;
        root.depth = 0;
        root.begin_token = 0;
        root.end_token = out.tokens.size();
        out.nodes.push_back(root);  // индекс 0

        std::size_t i = 0;
        while (i < out.tokens.size()) {
            std::size_t j = i + 1;
            while (j < out.tokens.size() && !out.tokens[j].starts_chunk) ++j;

            int head = -1;
            for (std::size_t k = i; k < j; ++k) {
                if (!out.tokens[k].function_word) head = static_cast<int>(k);
            }
            if (head < 0) head = static_cast<int>(i);

            // Агрегат чанка считается по сырому spansу, поэтому составные
            // термины внутри одной фразы срабатывают как обычно.
            const std::size_t chunk_begin = out.tokens[i].begin;
            const std::size_t chunk_end = out.tokens[j - 1].end;
            const MorphemeAnalysis chunk_analysis = compile_morpheme_analysis(
                text.substr(chunk_begin, chunk_end - chunk_begin), intent);

            SentenceNode chunk;
            chunk.parent = 0;
            chunk.depth = 1;
            chunk.begin_token = i;
            chunk.end_token = j;
            chunk.head_token = head;
            chunk.counts = chunk_analysis.cluster_counts;
            const int chunk_index = static_cast<int>(out.nodes.size());
            out.nodes.push_back(chunk);

            // Голова: собственный разбор токена плюс поддержка составного
            // термина, который заканчивается на этом слове.
            SentenceNode head_node;
            head_node.parent = chunk_index;
            head_node.depth = 2;
            head_node.begin_token = static_cast<std::size_t>(head);
            head_node.end_token = static_cast<std::size_t>(head) + 1;
            head_node.head_token = head;
            head_node.counts = out.tokens[static_cast<std::size_t>(head)]
                                   .analysis.cluster_counts;
            for (const CompoundRule& rule : kCompoundRules) {
                const int parts = rule.third.empty() ? 2 : 3;
                const int first = head - (parts - 1);
                if (first < static_cast<int>(i)) continue;
                const std::string_view rule_parts[3] = {
                    rule.first, rule.second, rule.third};
                bool matches = true;
                for (int k = 0; k < parts; ++k) {
                    const SentenceToken& token =
                        out.tokens[static_cast<std::size_t>(first + k)];
                    if (!token_equals(token.text, 0, token.text.size(),
                                      rule_parts[k])) {
                        matches = false;
                        break;
                    }
                }
                if (matches) {
                    head_node.counts[static_cast<std::size_t>(rule.primary)] +=
                        rule.weight;
                }
            }
            const int head_index = static_cast<int>(out.nodes.size());
            out.nodes.push_back(head_node);

            int chunk_depth = 2;
            for (std::size_t k = i; k < j; ++k) {
                if (static_cast<int>(k) == head) continue;
                SentenceNode dependent;
                dependent.parent = head_index;
                dependent.depth = 3;
                dependent.begin_token = k;
                dependent.end_token = k + 1;
                dependent.head_token = static_cast<int>(k);
                dependent.counts = out.tokens[k].analysis.cluster_counts;
                out.nodes.push_back(dependent);
                chunk_depth = 3;
            }
            if (chunk_depth > out.max_depth) out.max_depth = chunk_depth;

            for (int cluster = 0; cluster < 20; ++cluster) {
                const std::size_t c = static_cast<std::size_t>(cluster);
                out.head_counts[c] += head_node.counts[c];
                out.chunk_max_counts[c] =
                    std::max(out.chunk_max_counts[c],
                             out.nodes[static_cast<std::size_t>(chunk_index)]
                                 .counts[c]);
                out.root_counts[c] +=
                    out.nodes[static_cast<std::size_t>(chunk_index)].counts[c];
            }

            i = j;
        }

        out.nodes[0].counts = out.root_counts;

        // ------------------------------------------------------------ RDT слой
        // Локальная ротация поднимает голову чанка над самим контейнером.
        // Это не лексическое правило: решение следует из структуры дерева,
        // роли токена и массы его поддерева. Исходные parent/depth сохраняются
        // для обратимости, normalized_parent/depth образуют рабочее дерево.
        for (SentenceNode& node : out.nodes) {
            node.normalized_parent = node.parent;
            node.normalized_depth = node.depth;
        }

        for (std::size_t chunk_index = 1; chunk_index < out.nodes.size(); ++chunk_index) {
            SentenceNode& chunk = out.nodes[chunk_index];
            if (chunk.depth != 1) continue;

            int head_index = -1;
            for (std::size_t candidate = 1; candidate < out.nodes.size(); ++candidate) {
                const SentenceNode& child = out.nodes[candidate];
                if (child.normalized_parent == static_cast<int>(chunk_index) &&
                    child.head_token == chunk.head_token && child.depth == 2) {
                    head_index = static_cast<int>(candidate);
                    break;
                }
            }
            if (head_index < 0) continue;

            TreeRotation rotation;
            rotation.child = head_index;
            rotation.parent = static_cast<int>(chunk_index);
            rotation.pivot = chunk.normalized_parent;
            rotation.direction = 1;
            rotation.before_depth = out.nodes[static_cast<std::size_t>(head_index)]
                                        .normalized_depth;

            const int grandparent = chunk.normalized_parent;
            out.nodes[static_cast<std::size_t>(head_index)].normalized_parent = grandparent;
            chunk.normalized_parent = head_index;
            out.rotations.push_back(rotation);
        }

        for (std::size_t index = 0; index < out.nodes.size(); ++index) {
            int current = static_cast<int>(index);
            int depth = 0;
            int guard = 0;
            while (current >= 0 && guard++ <= static_cast<int>(out.nodes.size())) {
                const int parent = out.nodes[static_cast<std::size_t>(current)].normalized_parent;
                if (parent < 0) break;
                ++depth;
                current = parent;
            }
            out.nodes[index].normalized_depth = depth;
        }
        for (TreeRotation& rotation : out.rotations) {
            rotation.after_depth = out.nodes[static_cast<std::size_t>(rotation.child)]
                                       .normalized_depth;
        }
        out.normalized_root = 0;
        out.normalized_root_counts = out.root_counts;

        // Проверка ацикличности после серии ротаций. В штатном случае она
        // бесплатна относительно разбора и защищает графовое представление
        // от ошибочного поворота.
        for (std::size_t index = 0; index < out.nodes.size(); ++index) {
            int current = static_cast<int>(index);
            int guard = 0;
            while (current >= 0 && guard++ <= static_cast<int>(out.nodes.size())) {
                current = out.nodes[static_cast<std::size_t>(current)].normalized_parent;
            }
            if (guard > static_cast<int>(out.nodes.size())) {
                out.rotations.clear();
                for (SentenceNode& node : out.nodes) {
                    node.normalized_parent = node.parent;
                    node.normalized_depth = node.depth;
                }
                break;
            }
        }

        int normalized_max_depth = 0;
        for (const SentenceNode& node : out.nodes) {
            normalized_max_depth = std::max(normalized_max_depth,
                                            node.normalized_depth);
        }
        out.max_depth = std::max(out.max_depth, normalized_max_depth);
        return out;
    }

    std::pair<std::uint64_t, int> compile_morpheme_mask(
        std::string_view text, const std::array<float, 16>& intent) const noexcept {
        const MorphemeAnalysis analysis = compile_morpheme_analysis(text, intent);
        return {analysis.mask, analysis.matched_morphemes};
    }
};

} // namespace ae::mind
