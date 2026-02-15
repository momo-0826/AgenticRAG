import json
from langchain_core.documents import Document

def _load_json_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    return candidates

def _build_page_content(candidate: dict) -> str:
    skills_text = ", ".join(
        f"{s['name']}({s['experience_years']}年)" for s in candidate["skills"]
    )

    careers_text = "\n".join(
        f"- {c['company']}({c['period']}): {c['description']}"
        for c in candidate["careers"]
    )

    return (
        f"名前: {candidate['name']}\n"
        f"居住地: {candidate['location']}\n"
        f"職種: {candidate['job_title']}\n"
        f"スキル: {skills_text}\n"
        f"経歴:\n{careers_text}\n"
        f"希望時給: {candidate['hourly_rate']}円\n"
        f"稼働: 週{candidate['available_days_per_week']}日, "
        f"1日{candidate['available_hours_per_day']}時間"
    )

def _build_metadata(candidate: dict) -> dict:
    return {
        "id": candidate["id"],
        "name": candidate["name"],
        "location": candidate["location"],
        "job_title": candidate["job_title"],
        "skills": ",".join(s["name"] for s in candidate["skills"]),
        "skill_years": ",".join(
            f"{s['name']}:{s['experience_years']}" for s in candidate["skills"]
        ),
        "hourly_rate": candidate["hourly_rate"],
        "available_days_per_week": candidate["available_days_per_week"],
        "available_hours_per_day": candidate["available_hours_per_day"],
        "last_login": candidate["last_login"],
    }

def load_candidates(file_path: str) -> list[Document]:
    candidates = _load_json_file(file_path)
    
    documents = []
    for candidate in candidates:
        page_content = _build_page_content(candidate)
        metadata = _build_metadata(candidate)
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    return documents

