from typing import Dict, Tuple

import pandas as pd
import yaml


class RuleEngine:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def _job_level_filter(self, job_level: str, assessment_level: str) -> Tuple[bool, float]:
        jl_cfg = self.config.get("job_level", {})
        level_key = job_level.lower().strip() if job_level else ""
        level_rules = jl_cfg.get(level_key, {}) if isinstance(jl_cfg, dict) else {}
        exclude_levels = {lvl.lower().strip() for lvl in level_rules.get("exclude_levels", [])}
        weight = float(level_rules.get("weight", 0.5))  # Default weight if not found
        assessment_level_clean = assessment_level.lower().strip() if assessment_level else ""
        if assessment_level_clean in exclude_levels:
            return False, -1.0
        return True, weight

    def _hiring_stage_penalty(self, hiring_stage: str, duration_minutes: float) -> float:
        stage_cfg = self.config.get("hiring_stage", {})
        stage_key = hiring_stage.lower() if hiring_stage else ""
        cfg = stage_cfg.get(stage_key, {}) if isinstance(stage_cfg, dict) else {}
        threshold = cfg.get("duration_penalty_threshold")
        penalty = cfg.get("duration_penalty", 0.0)
        if threshold is None:
            return 0.0
        if duration_minutes and duration_minutes > threshold:
            return float(penalty)
        return 0.0

    def _skill_domain_boost(self, skills: Dict) -> float:
        domain_cfg = self.config.get("skill_domains", {})
        user_skills = {s.lower() for s in skills or []}
        boost_total = 0.0

        for domain, cfg in domain_cfg.items():
            boost_categories = {c.lower() for c in cfg.get("boost_categories", [])}
            boost = float(cfg.get("boost", 0.0))

            if domain == "technical" and any(keyword in skill for skill in user_skills for keyword in ["developer", "engineer", "technical"]):
                boost_total += boost
            if domain == "behavioral" and any(keyword in skill for skill in user_skills for keyword in ["communication", "leadership", "behavior"]):
                boost_total += boost
            if user_skills.intersection(boost_categories):
                boost_total += boost

        return boost_total

    def apply(self, job_req: Dict, df: pd.DataFrame) -> pd.DataFrame:
        base_weights = self.config.get("base_weights", {})
        pass_weight = float(base_weights.get("rule_pass", 1.0))
        fail_weight = float(base_weights.get("rule_fail", -0.5))

        results = []
        for _, row in df.iterrows():
            job_level_pass, job_level_score = self._job_level_filter(job_req.get("job_level"), row.get("job_level", ""))
            duration_penalty = self._hiring_stage_penalty(job_req.get("hiring_stage"), row.get("duration_minutes") or 0)
            domain_boost = self._skill_domain_boost(job_req.get("skills"))

            rule_pass = job_level_pass
            score = job_level_score + duration_penalty + domain_boost
            score += pass_weight if rule_pass else fail_weight

            results.append(
                {
                    "rule_pass": rule_pass,
                    "rule_score": score,
                    "rule_explanation": {
                        "job_level_pass": job_level_pass,
                        "job_level_score": job_level_score,
                        "duration_penalty": duration_penalty,
                        "domain_boost": domain_boost,
                    },
                }
            )

        enrich_df = df.copy()
        enrich_df[["rule_pass", "rule_score", "rule_explanation"]] = pd.DataFrame(results, index=df.index)
        return enrich_df
