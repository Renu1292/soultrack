from statistics import mean, median

from flask_login import current_user


def detect_burnout(sessions):
    """Detects burnout using mood trends + workload analysis
    """

    # CONSTANTS
    CRITICAL_MOODS = ["clueless", "frustrated", "exhausted"]
    WARNING_MOODS = ["meh", "slow_day"]
    MIN_SESSIONS = 3

    # VALIDATION
    if len(sessions) < MIN_SESSIONS:
        return False, "Not enough data to assess burnout."

    # Get recent data
    recent = sessions[-MIN_SESSIONS:]
    moods = [s.mood for s in recent]
    hours = [s.hours for s in recent]

    # Mood analysis
    critical = sum(m in CRITICAL_MOODS for m in moods)
    warning = sum(m in WARNING_MOODS for m in moods)

    # Hours analysis
    avg_hours = mean(hours)
    if len(sessions) > MIN_SESSIONS:
        baseline = mean(s.hours for s in sessions[-10:-MIN_SESSIONS]) if len(sessions) > 6 else mean(hours)
        overworked = avg_hours > 1.2 * baseline
    else:
        overworked = avg_hours > 0 # Fallback threshold

    # Detection logic
    if (critical >= 2 and overworked) or (warning >= 2 and avg_hours > 10):
        return True, "⚠️ Burnout warning: High workload with persistent low mood"

    return False, "✅ No significant burnout signs"


def suggest_weekly_goal(this_week_hours, past_week_hours=None):
    print("=== DEBUG START ===")
    print(f"Input: This Week:{this_week_hours}h, Past Weeks: {past_week_hours}")

    # Step 1 : Handle new users differently if they logged significant hours
    if not past_week_hours:
        if this_week_hours >= 30:
            print("New user but worked 30+ : Using 80% of this week as baseline")
            baseline = this_week_hours * 0.8
        else:
            baseline = 10
            print("New user with low hours : Default 10h baseline")
    else:
        # Step 2: Calculate baseline from last 1-3 weeks (median)
        valid_weeks = [h for h in past_week_hours if h > 0]
        if not valid_weeks:
            baseline = 10
            print("No valid past weeks - Default 10h")
        elif len(valid_weeks) == 1:
            baseline = (valid_weeks[0] + 10) / 2
            print(f"1 past week: Blending {valid_weeks[0]}h with default:({valid_weeks[0]} + 10)/2 = {baseline}h")
        elif len(valid_weeks) == 2:
            baseline = median(valid_weeks)
            print(f"2 past weeks: Average: ({valid_weeks[0]} +{valid_weeks[1]})/2 = {baseline}h")
        else:
            baseline = median(valid_weeks[-3:])
            print(f"3+ weeks: Median of {valid_weeks[-3:]}:{baseline}h")

    # Step 3: Calculate suggestion
    if this_week_hours >= baseline * 1.5:   # Worked 50% more than usual
        suggested = min(this_week_hours * 1.1, baseline * 1.5)
        print(f"Overworked ({this_week_hours}h >= {baseline*1.5}h : Cap at +10% or 1.5x baseline")
    else:
        suggested = (this_week_hours + baseline) / 2
        print(f"Normal week: Average of ({this_week_hours}h + {baseline}h)/2 = {suggested}h")

    # Apply safety limits
    final_goal = round(max(5, min(suggested, 50)))
    print(f"Final Suggested Goal: {final_goal}h")
    print("==DEBUG END==")
    return final_goal

def compute_productivity_score(total_hours, session_days, goal_target, goal_achieved, mood_list, burnout_flag,
                              baseline_hours=20):

    score = 50 # Neutral base
    breakdown = {}

    # Debugging prints
    print(f"Starting Score: {score}")
    print(f"Total Hours: {total_hours}, Session Days: {session_days}, Goal Target: {goal_target}, Goal Achieved: {goal_achieved}")
    print(f"Mood List: {mood_list}, Burnout Flag: {burnout_flag}")
    print(f"Baseline Hours: {baseline_hours}")

    # 1. Hour Quality (0-25pts)
    ideal_min = baseline_hours * 0.8
    ideal_max = baseline_hours * 1.2

    if total_hours == 0 and session_days == 0 and not mood_list:
        print("No data to score. Returning 0.")
        return 0, {}
    elif total_hours < ideal_min:
        hour_score = 5 + 5 * (total_hours / ideal_min) # Partial credit
    elif total_hours > ideal_max:
        hour_score = 25 - 10 * (total_hours / ideal_max) # Overwork penalty
    else:
        hour_score = 25 # Perfect Range

    breakdown["hours"] = round(hour_score, 2)
    score += hour_score
    print(f"Hour Score: {hour_score:2f} (Ideal Range: {ideal_min}-{ideal_max})")

    # 2. Consistency (0-15pts)
    # consistency_score = min(session_days * 1.5, 10) # Days
    if session_days >= 5:
        consistency_score = 15 # Full bonus
    else:
        consistency_score = session_days * 2

    breakdown["consistency"] = round(consistency_score, 2)
    score += consistency_score

    # 3. Goal Performance (0-20pts)
    if goal_target and goal_target > 0: # Explicit check for valid goal
        if goal_achieved:
            # Base 15 + scaled bonus (5 max for challenging goals)
            goal_bonus = min(5, (goal_target - baseline_hours)/5)
            goal_score = 15 + max(0, goal_bonus)
        else:
            # Partial credit based on progress
            goal_score = 10 * (total_hours / goal_target)
    else:
        goal_score = 0

    goal_score = min(20, max(0, round(goal_score, 2))) # Clamp to 0-20 range
    breakdown["goal"] = goal_score
    score += goal_score

    # Debug Prints
    print(f"Goal Debug: Target={goal_target}, Achieved={goal_achieved}")
    print(f"Calculated Goal Score: {goal_score:.2f}")

    # 4. Mood (0-20pts)
    mood_values = {
        'happy': +7, 'determined': +8, 'beast_mode': +9, 'accomplished': +8,
        'mind_blown': +7,  'calm': +6, 'meh': +5, 'slow_day': +5, 'clueless': -2,
        'frustrated': -3, 'exhausted': -2
    }
    mood_points = sum(mood_values.get(m, 0) for m in mood_list)
    max_possible = len(mood_list) * 10
    normalized_mood = (mood_points / max_possible) * 20 if max_possible else 0
    mood_score = max(0, normalized_mood) # Prevent negative score

    breakdown["mood"] = round(mood_score,2)
    score += mood_score

    print("Mood Debug:")
    print(" - Mood List:", mood_list)
    print(" - Raw Mood Points:", mood_points)
    print(" - Max Possible Mood Points:", max_possible)
    print(" - Normalised Mood Score (0-20):", round(normalized_mood, 2))

   # 5. Burnout (-15 to +5)
    if burnout_flag:
        burnout_score = -10 if len(mood_list) >= 5 else -5
    elif not burnout_flag and 'exhausted' not in mood_list:
        burnout_score = 4 # Recovery bonus
    else:
        burnout_score = 0

    breakdown["burnout"] = round(burnout_score, 2)
    score += burnout_score
    print(f"Burnout Score Adjustment: {burnout_score}")


    final_score = round(max(0, min(score, 100)), 2)
    print(f"Final Productivity Score: {final_score}\n")
    print(f"Final Breakdown -> Hours: {hour_score:.2f}, Consistency: {consistency_score:.2f}, Goal: {goal_score:.2f}, Mood: {normalized_mood:.2f}, Burnout: {burnout_score}")

    return final_score, breakdown












