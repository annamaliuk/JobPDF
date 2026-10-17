"""Fixed vocabulary for the synthetic ranking dataset (JM-25): role families, skills,
job titles and text templates.

Everything here is fictional or generic. Skills name real taxonomy concepts so the
dataset can run through JM-20 normalization: vacancies use the canonical name, CVs
often use an alias. Every alias listed here is a row of data/taxonomy/custom_skills.csv
(a test checks this), so normalization can resolve it.

Changing anything in this file changes the generated dataset; bump
``GENERATOR_VERSION`` in synthetic.py when you do.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillDef:
    slug: str  # ground-truth identity, "synthetic:<slug>" in truth keys
    name: str  # canonical name, as a vacancy writes it
    aliases: tuple[str, ...] = ()  # other spellings a CV may use


def _s(slug: str, name: str, *aliases: str) -> SkillDef:
    return SkillDef(slug, name, tuple(aliases))


SKILLS: dict[str, SkillDef] = {s.slug: s for s in [
    _s("python", "Python"),
    _s("sql", "SQL"),
    _s("airflow", "Apache Airflow", "airflow"),
    _s("spark", "Apache Spark", "spark", "pyspark"),
    _s("bigquery", "Google BigQuery", "bigquery", "gbq"),
    _s("dbt", "dbt", "data build tool"),
    _s("kafka", "Apache Kafka", "kafka"),
    _s("postgresql", "PostgreSQL", "postgres"),
    _s("docker", "Docker"),
    _s("terraform", "Terraform"),
    _s("go", "Go (programming language)", "golang", "go"),
    _s("nodejs", "Node.js", "node", "nodejs"),
    _s("redis", "Redis"),
    _s("rest-api", "REST API"),
    _s("grpc", "gRPC"),
    _s("rabbitmq", "RabbitMQ"),
    _s("django", "Django"),
    _s("mongodb", "MongoDB"),
    _s("javascript", "JavaScript", "js"),
    _s("typescript", "TypeScript", "ts"),
    _s("react", "React"),
    _s("redux", "Redux"),
    _s("vue", "Vue"),
    _s("angular", "Angular"),
    _s("html", "HTML", "html5"),
    _s("tailwind", "Tailwind"),
    _s("figma", "Figma"),
    _s("graphql", "GraphQL"),
    _s("machine-learning", "machine learning", "ml"),
    _s("nlp", "natural language processing", "nlp"),
    _s("pandas", "pandas"),
    _s("pytorch", "PyTorch"),
    _s("scikit-learn", "scikit-learn"),
    _s("kubernetes", "Kubernetes", "k8s"),
    _s("aws", "Amazon Web Services"),
    _s("azure", "Microsoft Azure", "azure"),
    _s("gcp", "Google Cloud Platform", "gcp", "google cloud"),
    _s("jenkins", "Jenkins"),
    _s("grafana", "Grafana"),
    _s("bash", "Bash"),
    _s("linux", "Linux"),
    _s("selenium", "Selenium"),
    _s("cypress", "Cypress"),
    _s("postman", "Postman"),
    _s("git", "Git"),
    _s("kotlin", "Kotlin"),
    _s("swift", "Swift"),
    _s("firebase", "Firebase"),
    _s("react-native", "React Native"),
    _s("miro", "Miro"),
    _s("amplitude", "Amplitude"),
    _s("google-analytics", "Google Analytics", "ga4"),
    _s("jira", "Jira"),
    _s("hypothesis-testing", "Hypothesis testing"),
    _s("market-research", "Market Research", "market analysis"),
    _s("wireframing", "Wireframing"),
    _s("notion", "Notion"),
]}

# Listed on CVs but never required, so they never change a grade.
SOFT_SKILLS = ("communication", "teamwork", "problem solving", "mentoring", "time management")
# Not in the taxonomy: they test that unrecognised skills are handled, not dropped.
UNKNOWN_SKILLS = ("Kanban boards", "internal tooling", "OKR planning")


@dataclass(frozen=True)
class Family:
    key: str
    core: tuple[str, ...]  # skill slugs a vacancy in this family requires from
    titles: tuple[str, ...]
    responsibilities: tuple[str, ...]  # vacancy lines
    cv_lines: tuple[str, ...]  # CV job description lines; {a} and {b} are skill names
    cv_lines_uk: tuple[str, ...] = ()  # Ukrainian CV lines, for a few candidates


FAMILIES: dict[str, Family] = {f.key: f for f in [
    Family(
        "data_engineering",
        ("python", "sql", "airflow", "spark", "bigquery", "dbt", "kafka", "postgresql",
         "docker", "terraform"),
        ("Data Engineer", "Analytics Engineer", "Data Platform Engineer"),
        ("Design and maintain batch and streaming data pipelines",
         "Model data in the cloud warehouse for analytics teams",
         "Improve reliability and freshness of reporting data",
         "Own data quality checks and pipeline monitoring",
         "Work with analysts to turn raw events into clean tables",
         "Migrate legacy ETL jobs to an orchestrated platform"),
        ("Built nightly ETL pipelines in {a} and {b} that load sales data into the warehouse.",
         "Moved event streams into the analytics warehouse using {a}; cut data delays to minutes.",
         "Maintained {a} jobs and wrote data quality checks with {b}.",
         "Modelled reporting tables in {a} for the finance and marketing teams."),
        ("Будував нічні ETL-конвеєри на {a} та {b}, що завантажують дані у сховище.",
         "Підтримував задачі {a} і писав перевірки якості даних на {b}."),
    ),
    Family(
        "backend",
        ("python", "go", "nodejs", "postgresql", "redis", "rest-api", "grpc", "docker",
         "rabbitmq", "django", "mongodb", "kafka"),
        ("Backend Engineer", "Backend Developer", "API Engineer"),
        ("Design and build REST and gRPC services for the core product",
         "Own the data model and database performance of key services",
         "Write well-tested server code and review teammates' changes",
         "Integrate payment and messaging providers",
         "Improve latency and reliability of public APIs",
         "Break a monolith into smaller services"),
        ("Built and ran REST services in {a} backed by {b} for an online store.",
         "Designed the order service with {a}; reduced p95 latency by a third.",
         "Wrote background workers with {a} and {b} for billing and email.",
         "Owned the public API, its database schema and its {a} integration."),
        ("Розробляв REST-сервіси на {a} з базою {b} для інтернет-магазину.",
         "Писав фонові обробники на {a} та {b} для білінгу та розсилок."),
    ),
    Family(
        "frontend",
        ("javascript", "typescript", "react", "redux", "vue", "angular", "html", "tailwind",
         "figma", "graphql"),
        ("Frontend Engineer", "Frontend Developer", "UI Engineer"),
        ("Build responsive web interfaces from design mockups",
         "Own the component library and its accessibility",
         "Work with designers and backend engineers on new features",
         "Improve page load speed and Core Web Vitals",
         "Write unit and end-to-end tests for the web app"),
        ("Built the customer dashboard in {a} and {b} from Figma mockups.",
         "Rewrote the checkout pages with {a}; improved load time by 40%.",
         "Maintained a shared component library in {a} used by four teams.",
         "Implemented an accessible design system with {a} and {b}."),
        ("Створив клієнтську панель на {a} та {b} за макетами дизайнерів.",
         "Підтримував спільну бібліотеку компонентів на {a}."),
    ),
    Family(
        "ml",
        ("python", "machine-learning", "nlp", "pandas", "sql", "pytorch", "scikit-learn",
         "docker", "spark"),
        ("Machine Learning Engineer", "Data Scientist", "NLP Engineer"),
        ("Train and evaluate models for recommendation and search",
         "Turn prototypes into production model services",
         "Design experiments and measure model impact",
         "Build feature pipelines and keep training data clean",
         "Monitor models in production for drift"),
        ("Trained ranking models with {a} and {b} for product search.",
         "Built a text classification service using {a}; raised precision to 0.91.",
         "Prepared training data with {a} and evaluated models with {b}.",
         "Shipped a churn model to production and monitored it for drift."),
    ),
    Family(
        "devops",
        ("kubernetes", "docker", "terraform", "aws", "jenkins", "grafana", "bash", "linux",
         "azure", "gcp"),
        ("DevOps Engineer", "Site Reliability Engineer", "Platform Engineer"),
        ("Run and scale Kubernetes clusters for production services",
         "Manage cloud infrastructure as code",
         "Build CI/CD pipelines for all product teams",
         "Own monitoring, alerting and incident response",
         "Reduce cloud costs without hurting reliability"),
        ("Ran production clusters on {a} and managed infrastructure with {b}.",
         "Built CI/CD pipelines in {a} for twelve services.",
         "Set up monitoring and alerting with {a}; halved incident response time.",
         "Moved services to {a} and wrote the deployment scripts in {b}."),
    ),
    Family(
        "qa",
        ("selenium", "cypress", "postman", "python", "javascript", "rest-api", "sql",
         "jenkins", "git"),
        ("QA Automation Engineer", "Test Engineer", "Quality Engineer"),
        ("Build and maintain automated UI and API test suites",
         "Plan test coverage for new features with developers",
         "Run regression tests in the CI pipeline",
         "Report and track defects until they are fixed",
         "Improve test stability and speed"),
        ("Automated regression tests with {a} and {b} for a banking web app.",
         "Wrote API tests in {a}; caught most defects before release.",
         "Ran nightly test suites in {a} and reported results to the team.",
         "Planned test coverage for new features and tracked defects to closure."),
    ),
    Family(
        "mobile",
        ("kotlin", "swift", "firebase", "react-native", "rest-api", "git", "figma",
         "graphql"),
        ("Mobile Developer", "Android Developer", "iOS Developer"),
        ("Build and ship features in our iOS and Android apps",
         "Work with designers on smooth, accessible mobile screens",
         "Integrate the apps with backend APIs and push notifications",
         "Improve app start time and crash-free rate",
         "Publish releases to the App Store and Google Play"),
        ("Built Android features in {a} and connected them to {b}.",
         "Shipped an iOS booking app with {a}; crash-free sessions above 99%.",
         "Added push notifications and analytics with {a}.",
         "Maintained a cross-platform app in {a} for 200k users."),
    ),
    Family(
        "product",
        ("figma", "miro", "amplitude", "google-analytics", "sql", "jira",
         "hypothesis-testing", "market-research", "wireframing", "notion"),
        ("Product Manager", "Product Owner", "Product Analyst"),
        ("Own the roadmap for one product area",
         "Talk to users and turn findings into clear requirements",
         "Define success metrics and run experiments",
         "Work with design and engineering from idea to release",
         "Report product performance to leadership"),
        ("Owned the onboarding roadmap; ran experiments tracked in {a}.",
         "Wrote requirements in {a} and worked with designers in {b}.",
         "Ran user interviews and market research for a new pricing plan.",
         "Defined product metrics and built dashboards with {a}."),
    ),
]}

# Families whose work overlaps enough that a move between them is plausible.
ADJACENT: frozenset[frozenset[str]] = frozenset(frozenset(p) for p in [
    ("data_engineering", "backend"),
    ("data_engineering", "ml"),
    ("backend", "devops"),
    ("backend", "frontend"),
    ("backend", "mobile"),
    ("backend", "qa"),
    ("frontend", "mobile"),
    ("frontend", "qa"),
    ("frontend", "product"),
])

COMPANIES = (
    "Acme Widgets", "Bluefin Labs", "Northwind Apps", "Cobalt Systems", "Lumen Data",
    "Pinecrest Software", "Harbor Analytics", "Quartz Retail", "Juniper Health",
    "Orbit Logistics", "Maple Payments", "Tidewater Media",
)
LOCATIONS = ("Worldwide", "Europe only", "USA only", "UTC-2 to UTC+4", "Ukraine, Poland")
SENIORITY = ("Junior", "", "Senior")
UNIVERSITIES = ("Example State University", "Northfield Technical College",
                "Riverside University")
DEGREES = ("BSc", "MSc")
