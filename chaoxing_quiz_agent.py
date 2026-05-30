#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import getpass
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
SUPERSTAR_DIR = ROOT_DIR / "SuperStar"
LOCAL_CONFIG_PATH = ROOT_DIR / "chaoxing_quiz_agent.local.json"
DEFAULT_OPENAI_BASE_URL = "https://api.siliconflow.cn/v1"

if str(SUPERSTAR_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERSTAR_DIR))


TRUE_WORDS = {
    "true",
    "t",
    "yes",
    "y",
    "1",
    "正确",
    "对",
    "是",
    "√",
}
FALSE_WORDS = {
    "false",
    "f",
    "no",
    "n",
    "0",
    "错误",
    "错",
    "否",
    "非",
    "×",
}


def load_superstar_modules():
    from api.base import Account, Chaoxing, SessionManager  # type: ignore
    from api.decode import decode_questions_info  # type: ignore

    return Account, Chaoxing, SessionManager, decode_questions_info


def load_local_config() -> Dict[str, str]:
    if not LOCAL_CONFIG_PATH.exists():
        return {}
    try:
        with LOCAL_CONFIG_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="超星学习通答题脚本")
    parser.add_argument(
        "target",
        help="答题页 URL 或课程码(courseId)",
    )
    parser.add_argument("--tiku", help="本地题库 JSON 文件路径")
    parser.add_argument("--username", help="超星账号，用于 cookies 失效时重新登录")
    parser.add_argument("--password", help="超星密码，用于 cookies 失效时重新登录")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", help="LLM 模型名")
    parser.add_argument("--api-key", help="OpenAI 兼容接口 API Key；未传则读取 OPENAI_API_KEY")
    parser.add_argument(
        "--base-url",
        help="OpenAI 兼容接口 Base URL；未传则依次读取本地配置、OPENAI_BASE_URL，默认 https://api.siliconflow.cn/v1",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "url", "course"],
        default="auto",
        help="目标类型：自动识别、URL 直答、课程码自动找答题任务",
    )
    parser.add_argument(
        "--course-name",
        help="当课程码不好确定时，可传课程名称关键字辅助匹配",
    )
    parser.add_argument(
        "--include-finished",
        action="store_true",
        help="遍历课程时包含已完成章节，默认跳过已完成章节",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="低于该覆盖率时只保存不提交，范围 0-1",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="强制提交；否则命中率不足时只保存",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="只保存不提交",
    )
    parser.add_argument(
        "--allow-blank-submit",
        action="store_true",
        help="允许带空答案提交；默认只在保存模式下保留空题",
    )
    return parser.parse_args()


def check_dependencies(use_llm: bool) -> None:
    missing: List[str] = []
    for module_name, package_name in [
        ("bs4", "beautifulsoup4"),
        ("lxml", "lxml"),
        ("loguru", "loguru"),
        ("pyaes", "pyaes"),
        ("fontTools", "fonttools"),
        ("httpx", "httpx"),
        ("tqdm", "tqdm"),
    ]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if use_llm:
        try:
            __import__("openai")
        except ImportError:
            missing.append("openai")

    if missing:
        print("[错误] 缺少依赖，请先安装：")
        print("  pip install " + " ".join(sorted(set(missing))))
        sys.exit(1)


def get_api_key(args: Optional[argparse.Namespace] = None) -> str:
    if args and args.api_key:
        return args.api_key
    local_config = load_local_config()
    if local_config.get("api_key"):
        return str(local_config["api_key"])
    return os.environ.get("OPENAI_API_KEY", "")


def get_base_url(args: Optional[argparse.Namespace] = None) -> str:
    if args and args.base_url:
        return args.base_url
    local_config = load_local_config()
    if local_config.get("base_url"):
        return str(local_config["base_url"])
    return os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_answer_text(answer: str) -> List[str]:
    value = str(answer or "").strip()
    if not value:
        return []
    for sep in ["\n", "###", "##", "|", ";", "；", ",", "，", "、", "/", "\\"]:
        if sep in value:
            parts = [clean_text(part) for part in value.split(sep)]
            parts = [part for part in parts if part]
            if parts:
                return parts
    return [value]


def normalize_option_text(option_text: str) -> str:
    text = clean_text(option_text)
    return re.sub(r"^[A-HＡ-Ｈ][\.\s、:：）)]*", "", text).strip()


def load_json_file(path: str) -> object:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_url_target(target: str) -> bool:
    return target.startswith("http://") or target.startswith("https://")


class LocalTikuSolver:
    def __init__(self, tiku_path: str):
        self.tiku_path = tiku_path
        self.tiku = self._load_tiku()

    def _load_tiku(self) -> List[Dict[str, str]]:
        if not os.path.exists(self.tiku_path):
            raise FileNotFoundError(f"题库文件不存在: {self.tiku_path}")

        data = load_json_file(self.tiku_path)
        if isinstance(data, dict):
            return [{"question": key, "answer": value} for key, value in data.items()]
        if isinstance(data, list):
            normalized = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                question = item.get("question") or item.get("title") or item.get("topic")
                answer = item.get("answer") or item.get("answers")
                if question is None or answer is None:
                    continue
                normalized.append({"question": str(question), "answer": answer})
            return normalized
        raise ValueError("题库格式不支持，应为 JSON 对象或数组")

    def _find_best_match(self, question: str) -> Optional[Dict[str, str]]:
        target = clean_text(question)
        best_ratio = 0.0
        best_item: Optional[Dict[str, str]] = None

        for item in self.tiku:
            candidate = clean_text(item.get("question", ""))
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item

        if best_ratio >= 0.84:
            return best_item
        return None

    def solve(self, q_info: Dict[str, str]) -> Optional[str]:
        item = self._find_best_match(q_info["title"])
        if not item:
            return None
        return stringify_answer(item.get("answer"))


class LlmSolver:
    def __init__(self, model: str, api_key: str, base_url: str):
        import openai

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.api_key = api_key

    def solve(self, q_info: Dict[str, str]) -> Optional[str]:
        if not self.api_key:
            return None

        option_lines = q_info["options"].splitlines() if q_info.get("options") else []
        options_text = "\n".join(option_lines)
        q_type = q_info["type"]

        type_prompt = {
            "single": "这是单选题，请返回唯一正确选项内容。",
            "multiple": "这是多选题，请返回所有正确选项内容。",
            "judgement": "这是判断题，请只返回“正确”或“错误”。",
            "completion": "这是填空题，请返回填空答案。",
            "shortanswer": "这是简答题，请返回简短准确答案。",
        }.get(q_type, "请返回正确答案。")

        user_prompt = (
            f"{type_prompt}\n"
            f"题目：{q_info['title']}\n"
            f"选项：\n{options_text}\n\n"
            "严格输出 JSON，格式为 {\"Answer\":[\"答案1\",\"答案2\"]}，不要输出额外文字。"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": "你是超星学习通答题助手，只输出 JSON。"},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            print(f"[LLM] 请求失败: {exc}")
            return None

        content = (response.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            print(f"[LLM] 返回内容不是有效 JSON: {content[:200]}")
            return None

        answers = parsed.get("Answer") or parsed.get("answer")
        return stringify_answer(answers)


class CombinedSolver:
    def __init__(self, local_solver: Optional[LocalTikuSolver], llm_solver: Optional[LlmSolver]):
        self.local_solver = local_solver
        self.llm_solver = llm_solver

    def solve(self, q_info: Dict[str, str]) -> tuple[Optional[str], str]:
        if self.local_solver:
            answer = self.local_solver.solve(q_info)
            if answer:
                return answer, "local_tiku"

        if self.llm_solver:
            answer = self.llm_solver.solve(q_info)
            if answer:
                return answer, "llm"

        return None, "none"


def stringify_answer(answer: object) -> Optional[str]:
    if answer is None:
        return None
    if isinstance(answer, list):
        parts = [clean_text(str(item)) for item in answer if clean_text(str(item))]
        return "\n".join(parts) if parts else None
    value = clean_text(str(answer))
    return value or None


def parse_options(options_text: str) -> List[Dict[str, str]]:
    parsed: List[Dict[str, str]] = []
    for raw in options_text.splitlines():
        line = clean_text(raw)
        if not line:
            continue
        match = re.match(r"^([A-HＡ-Ｈ])[\.\s、:：）)]*(.*)$", line)
        if match:
            label = match.group(1).upper()
            text = clean_text(match.group(2))
        else:
            label = chr(ord("A") + len(parsed))
            text = line
        parsed.append({"label": label, "text": text, "raw": line})
    return parsed


def is_subsequence(needle: str, haystack: str) -> bool:
    needle = clean_text(needle)
    haystack = clean_text(haystack)
    if not needle or not haystack:
        return False
    return needle in haystack or haystack in needle


def map_answer_to_submit_value(q_info: Dict[str, str], raw_answer: str) -> Optional[str]:
    q_type = q_info["type"]
    answer = clean_text(raw_answer)
    if not answer:
        return None

    if q_type == "judgement":
        if re.fullmatch(r"[A-Ha-h]", answer):
            options = parse_options(q_info.get("options", ""))
            option = next((opt for opt in options if opt["label"] == answer.upper()), None)
            if option:
                option_text = option["text"].lower()
                if option_text in TRUE_WORDS or any(word in option["text"] for word in ["正确", "对", "是", "√"]):
                    return "true"
                if option_text in FALSE_WORDS or any(word in option["text"] for word in ["错误", "错", "否", "非", "×"]):
                    return "false"
        normalized = answer.lower()
        if normalized in TRUE_WORDS:
            return "true"
        if normalized in FALSE_WORDS:
            return "false"
        return None

    if q_type in {"completion", "shortanswer", "unknown"}:
        return raw_answer.strip()

    options = parse_options(q_info.get("options", ""))
    if not options:
        return None

    if re.fullmatch(r"[A-Ha-h,\s，、/\\]+", answer):
        labels = re.findall(r"[A-H]", answer.upper())
        if q_type == "single":
            return labels[0] if labels else None
        return "".join(sorted(dict.fromkeys(labels)))

    result_labels: List[str] = []
    answers = split_answer_text(raw_answer)
    for answer_item in answers:
        normalized_answer = normalize_option_text(answer_item)
        if not normalized_answer:
            continue

        exact = next(
            (opt for opt in options if normalize_option_text(opt["text"]) == normalized_answer),
            None,
        )
        if exact:
            result_labels.append(exact["label"])
            continue

        contain = next(
            (
                opt
                for opt in options
                if is_subsequence(normalized_answer, normalize_option_text(opt["text"]))
            ),
            None,
        )
        if contain:
            result_labels.append(contain["label"])
            continue

        best_ratio = 0.0
        best_option: Optional[Dict[str, str]] = None
        for option in options:
            ratio = difflib.SequenceMatcher(
                None,
                normalized_answer,
                normalize_option_text(option["text"]),
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_option = option
        if best_option and best_ratio >= 0.78:
            result_labels.append(best_option["label"])

    if not result_labels:
        return None

    deduped = "".join(sorted(dict.fromkeys(result_labels)))
    if q_type == "single":
        return deduped[:1]
    return deduped


class ChaoxingQuizAgent:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        Account, Chaoxing, SessionManager, decode_questions_info = load_superstar_modules()
        self._SessionManager = SessionManager
        self._decode_questions_info = decode_questions_info
        self.api_key = get_api_key(args)
        self.base_url = get_base_url(args)

        local_solver = LocalTikuSolver(args.tiku) if args.tiku else None
        llm_solver = LlmSolver(args.model, self.api_key, self.base_url) if self.api_key else None
        if not local_solver and not llm_solver:
            raise RuntimeError("未提供题库，且未配置 AI。请传 --tiku 或 --api-key（或设置 OPENAI_API_KEY）")

        self.chaoxing = Chaoxing(
            account=Account(args.username or "", args.password or "")
        )
        self.session = None
        self.solver = CombinedSolver(local_solver=local_solver, llm_solver=llm_solver)

    def _set_account(self, username: str, password: str) -> None:
        self.args.username = clean_text(username)
        self.args.password = password
        self.chaoxing.account.username = self.args.username
        self.chaoxing.account.password = self.args.password

    def _prompt_for_credentials(self) -> bool:
        username = clean_text(self.args.username or "")
        password = self.args.password or ""

        if not username:
            try:
                username = clean_text(input("请输入超星账号: "))
            except EOFError:
                return False
        if not username:
            return False

        if not password:
            try:
                password = getpass.getpass("请输入超星密码: ")
            except EOFError:
                return False
        if not password:
            return False

        self._set_account(username, password)
        return True

    def login(self) -> None:
        if self.args.username and self.args.password:
            result = self.chaoxing.login(login_with_cookies=False)
            if result.get("status"):
                print("[登录] 已通过账号密码登录")
                self.session = self._SessionManager.get_session()
                return
            print(f"[登录] 账号密码登录失败，将尝试 cookies: {result.get('msg', '未知错误')}")

        result = self.chaoxing.login(login_with_cookies=True)
        if result.get("status"):
            print("[登录] 已通过 cookies 登录")
            self.session = self._SessionManager.get_session()
            return

        print(f"[登录] cookies 登录失败: {result.get('msg', '未知错误')}")

        if self._prompt_for_credentials():
            result = self.chaoxing.login(login_with_cookies=False)
            if result.get("status"):
                print("[登录] 已通过交互输入的账号密码登录")
                self.session = self._SessionManager.get_session()
                return

        message = result.get("msg", "登录失败")
        raise RuntimeError(f"无法登录超星: {message}")

    def fetch_work_page(self, url: str) -> tuple[str, str]:
        assert self.session is not None

        response = self.session.get(url, allow_redirects=True)
        final_url = response.url
        html = response.text

        if response.status_code != 200:
            raise RuntimeError(f"打开答题页失败: HTTP {response.status_code}")
        if "passport2.chaoxing.com" in final_url or "登录" in html[:500]:
            raise RuntimeError("当前会话未登录，跳转到了登录页")
        if "<form" not in html or "singleQuesId" not in html:
            raise RuntimeError("当前页面不是可解析的答题页，请直接传作业/章节测验页面 URL")

        return final_url, html

    def find_course(self) -> Dict[str, str]:
        all_courses = self.chaoxing.get_course_list()
        target = clean_text(self.args.target)
        course_name = clean_text(self.args.course_name or "")

        matched = [
            course
            for course in all_courses
            if course["courseId"] == target
        ]
        if len(matched) == 1:
            return matched[0]

        fuzzy_matches = []
        for course in all_courses:
            haystacks = [
                clean_text(course.get("title", "")),
                clean_text(course.get("desc", "")),
            ]
            if target and any(target in item for item in haystacks if item):
                fuzzy_matches.append(course)
                continue
            if course_name and any(course_name in item for item in haystacks if item):
                fuzzy_matches.append(course)

        deduped: List[Dict[str, str]] = []
        seen_ids = set()
        for course in fuzzy_matches:
            if course["courseId"] in seen_ids:
                continue
            seen_ids.add(course["courseId"])
            deduped.append(course)

        if len(deduped) == 1:
            return deduped[0]

        if not deduped:
            raise RuntimeError(f"未找到课程: {target}")

        lines = [f"{course['courseId']} | {course['title']}" for course in deduped[:10]]
        raise RuntimeError("匹配到多个课程，请改用更精确的课程码:\n" + "\n".join(lines))

    def fetch_work_page_from_job(
        self,
        course: Dict[str, str],
        point: Dict[str, str],
        job_info: Dict[str, str],
        job: Dict[str, str],
    ) -> Tuple[str, str]:
        assert self.session is not None

        response = self.session.get(
            "https://mooc1.chaoxing.com/mooc-ans/api/work",
            params={
                "api": "1",
                "workId": job["jobid"].replace("work-", ""),
                "jobid": job["jobid"],
                "originJobId": job["jobid"],
                "needRedirect": "true",
                "skipHeader": "true",
                "knowledgeid": str(point["id"]),
                "ktoken": job_info["ktoken"],
                "cpi": course["cpi"],
                "ut": "s",
                "clazzId": course["clazzId"],
                "type": "",
                "enc": job["enc"],
                "mooc2": "1",
                "courseid": course["courseId"],
            },
            allow_redirects=True,
        )

        if response.status_code != 200:
            raise RuntimeError(f"打开任务失败: HTTP {response.status_code}")

        html = response.text
        final_url = str(response.url)
        if "教师未创建完成该测验" in html:
            raise RuntimeError("教师未创建完成该测验")
        if "<form" not in html or "singleQuesId" not in html:
            raise RuntimeError("返回页面不是可解析的答题页")

        return final_url, html

    def resolve_targets(self) -> List[Tuple[str, str]]:
        mode = self.args.mode
        target = self.args.target

        if mode == "url" or (mode == "auto" and is_url_target(target)):
            return [("URL", target)]

        course = self.find_course()
        print(f"[课程] {course['courseId']} | {course['title']}")

        point_list = self.chaoxing.get_course_point(course["courseId"], course["clazzId"], course["cpi"])
        work_targets: List[Tuple[str, str]] = []
        seen_urls = set()

        for point in point_list.get("points", []):
            if point.get("has_finished") and not self.args.include_finished:
                continue

            jobs, job_info = self.chaoxing.get_job_list(course, point)
            if job_info.get("notOpen", False):
                continue

            for job in jobs:
                if job.get("type") != "workid":
                    continue
                try:
                    work_url, _html = self.fetch_work_page_from_job(course, point, job_info, job)
                except Exception as exc:
                    print(f"[跳过] {point['title']} | {exc}")
                    continue
                if work_url in seen_urls:
                    continue
                seen_urls.add(work_url)
                label = f"{point['title']} | {job['jobid']}"
                work_targets.append((label, work_url))

        if not work_targets:
            raise RuntimeError("该课程下未找到可答题的作业/测验任务")

        print(f"[发现] 共找到 {len(work_targets)} 个答题任务")
        return work_targets

    def solve_questions(self, parsed: Dict[str, object]) -> tuple[Dict[str, object], List[Dict[str, str]], float]:
        questions: List[Dict[str, str]] = parsed["questions"]  # type: ignore[assignment]
        solved_rows: List[Dict[str, str]] = []
        found_answers = 0

        for index, question in enumerate(questions, start=1):
            title = clean_text(question["title"])
            raw_answer, source = self.solver.solve(question)
            submit_value = map_answer_to_submit_value(question, raw_answer or "")

            if submit_value:
                found_answers += 1
                question[f"answerSource{question['id']}"] = "cover"
                question["answerField"][f"answer{question['id']}"] = submit_value
            else:
                question[f"answerSource{question['id']}"] = "blank"
                question["answerField"][f"answer{question['id']}"] = ""

            solved_rows.append(
                {
                    "index": str(index),
                    "type": question["type"],
                    "title": title,
                    "source": source,
                    "raw_answer": raw_answer or "",
                    "submit_value": submit_value or "",
                }
            )

        total_questions = len(questions)
        coverage = (found_answers / total_questions) if total_questions else 0.0
        return parsed, solved_rows, coverage

    def build_submit_payload(self, parsed: Dict[str, object], coverage: float) -> Dict[str, object]:
        questions: List[Dict[str, str]] = parsed["questions"]  # type: ignore[assignment]

        if self.args.save_only:
            py_flag = "1"
        elif self.args.submit:
            py_flag = ""
        elif coverage >= self.args.min_coverage:
            py_flag = ""
        else:
            py_flag = "1"

        payload = dict(parsed)
        payload["pyFlag"] = py_flag

        for question in questions:
            answer_key = f"answer{question['id']}"
            answer_type_key = f"answertype{question['id']}"
            answer_value = question["answerField"][answer_key]

            if py_flag == "1":
                payload[answer_key] = answer_value
            else:
                if answer_value or self.args.allow_blank_submit:
                    payload[answer_key] = answer_value
                else:
                    raise RuntimeError(
                        f"存在未答题目且当前为提交模式，题目：{question['title'][:80]}"
                    )

            payload[answer_type_key] = question["answerField"][answer_type_key]

        del payload["questions"]
        return payload

    def submit(self, work_url: str, payload: Dict[str, object]) -> Dict[str, object]:
        assert self.session is not None

        response = self.session.post(
            "https://mooc1.chaoxing.com/mooc-ans/work/addStudentWorkNew",
            data=payload,
            headers={
                "Origin": "https://mooc1.chaoxing.com",
                "Referer": work_url,
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"提交失败: HTTP {response.status_code} {response.text[:200]}")

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"提交返回不是 JSON: {response.text[:200]}") from exc

    def run_single_target(self, label: str, target_url: str) -> None:
        work_url, html = self.fetch_work_page(target_url)
        parsed = self._decode_questions_info(html)

        if not parsed.get("questions"):
            raise RuntimeError("未解析到题目")

        parsed, solved_rows, coverage = self.solve_questions(parsed)

        print(f"[任务] {label}")
        print(f"[题目] 共 {len(solved_rows)} 题")
        for row in solved_rows:
            print(
                f"[{row['index']}] {row['type']} | {row['source']} | "
                f"{row['submit_value'] or '未命中'} | {row['title'][:80]}"
            )

        action = "提交"
        if self.args.save_only or (not self.args.submit and coverage < self.args.min_coverage):
            action = "保存"
        print(f"[覆盖率] {coverage:.0%}，将执行：{action}")

        payload = self.build_submit_payload(parsed, coverage)
        result = self.submit(work_url, payload)

        if result.get("status"):
            print(f"[成功] {result.get('msg', action + '成功')}")
        else:
            raise RuntimeError(f"{action}失败: {result}")

    def run(self) -> None:
        self.login()
        if self.args.tiku:
            print("[答题源] 本地题库优先，未命中时自动回退 AI")
        else:
            print("[答题源] 未提供题库，将直接使用 AI 答题")
        targets = self.resolve_targets()

        failures = 0
        for index, (label, target_url) in enumerate(targets, start=1):
            print(f"\n{'=' * 60}")
            print(f"[进度] {index}/{len(targets)}")
            print(f"{'=' * 60}")
            try:
                self.run_single_target(label, target_url)
            except Exception as exc:
                failures += 1
                print(f"[失败] {label} | {exc}")

        if failures:
            raise RuntimeError(f"任务结束，失败 {failures} 个")


def main() -> None:
    args = parse_args()
    check_dependencies(use_llm=bool(get_api_key(args)))

    try:
        agent = ChaoxingQuizAgent(args)
        agent.run()
    except Exception as exc:
        print(f"[错误] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
