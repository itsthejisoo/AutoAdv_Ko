from logging_utils import log as _log


def log(message, *args, **kwargs):
    return _log(f"PromptEnhancer: {message}", *args, **kwargs)


def enhance_prompt_with_patterns(base_prompt, pattern_manager, target_model=None, prompt_type="initial"):
    try:
        turn_number = 1 if prompt_type == "initial" else None
        pattern_hints = pattern_manager.generate_system_prompt_hints(
            target_model=target_model,
            turn_number=turn_number,
        )

        model_insights = ""
        if target_model:
            model_insights = generate_model_specific_insights(pattern_manager, target_model)

        enhancement = ""

        if pattern_hints:
            enhancement += f"\n\n학습된 성공 기법:\n{pattern_hints}"

        if model_insights:
            enhancement += f"\n\n{target_model} 모델 특화 인사이트:\n{model_insights}"

        if prompt_type == "followup":
            recent_examples = get_recent_successful_examples(pattern_manager)
            if recent_examples:
                enhancement += f"\n\n최근 성공 패턴:\n{recent_examples}"

        return base_prompt + enhancement
    except Exception as e:
        log(f"Error enhancing prompt with patterns: {e}", "warning")
        return base_prompt


def generate_model_specific_insights(pattern_manager, target_model):
    try:
        model_patterns = pattern_manager.patterns.get("success_by_model", {})
        if target_model not in model_patterns:
            return ""

        model_data = model_patterns[target_model]
        insights = []

        if isinstance(model_data, dict):
            if "success_rate" in model_data:
                rate = model_data["success_rate"]
                insights.append(f"- {target_model} 대상 성공률: {rate:.1%}")

            if "effective_techniques" in model_data:
                techniques = model_data["effective_techniques"]
                top_techniques = sorted(techniques.items(), key=lambda x: x[1], reverse=True)[:3]
                if top_techniques:
                    insights.append("- 가장 효과적인 기법:")
                    for technique, count in top_techniques:
                        insights.append(f"  - {technique.replace('_', ' ').title()}: {count}회 성공")
        else:
            insights.append(f"- {target_model} 대상 총 성공 횟수: {model_data}")

        return "\n".join(insights) if insights else ""
    except Exception as e:
        log(f"Error generating model insights: {e}", "debug")
        return ""


def generate_temperature_insights(pattern_manager):
    try:
        effective_prompts = pattern_manager.patterns.get("effective_prompts", [])
        if not effective_prompts:
            return ""

        temps = []
        for prompt_data in effective_prompts:
            if isinstance(prompt_data, dict) and "temperature" in prompt_data:
                temp_value = prompt_data["temperature"]
                if isinstance(temp_value, (int, float)):
                    temps.append(temp_value)

        if not temps:
            return ""

        avg_temp = sum(temps) / len(temps)
        min_temp = min(temps)
        max_temp = max(temps)

        insights = [
            f"- 평균 성공 온도: {avg_temp:.2f}",
            f"- 효과적 범위: {min_temp:.2f} - {max_temp:.2f}",
        ]

        if avg_temp < 0.5:
            insights.append("- 권장: 낮은 온도(더 집중적)가 효과적인 경향")
        elif avg_temp > 1.0:
            insights.append("- 권장: 높은 온도(더 창의적)가 효과적인 경향")
        else:
            insights.append("- 권장: 중간 온도가 효과적")

        return "\n".join(insights)
    except Exception as e:
        log(f"Error generating temperature insights: {e}", "debug")
        return ""


def get_recent_successful_examples(pattern_manager, limit=2):
    try:
        effective_prompts = pattern_manager.patterns.get("effective_prompts", [])
        if not effective_prompts:
            return ""

        recent_prompts = effective_prompts[-limit:] if len(effective_prompts) >= limit else effective_prompts

        examples = []
        for i, prompt_data in enumerate(recent_prompts, 1):
            if isinstance(prompt_data, dict):
                prompt_text = prompt_data.get("prompt", "")
                score = prompt_data.get("evaluation_score", 0)
                techniques = prompt_data.get("techniques", [])

                if len(prompt_text) > 200:
                    prompt_text = prompt_text[:200] + "..."

                example = (
                    f"{i}. **Score: {score:.2f}** | Techniques: {', '.join(techniques[:3])}\n"
                    f"   \"{prompt_text}\""
                )
                examples.append(example)

        return "\n".join(examples) if examples else ""
    except Exception as e:
        log(f"Error getting recent examples: {e}", "debug")
        return ""
