GPQA_cot = "Please show your choice in the answer field with only the choice letter, e.g.,\"ANSWER\": \"C\"."
MATH_cot = "Please reason step by step, and put your final answer within \\boxed{}."

GPQA_retry = "Please show your choice in the answer field with only the choice letter, e.g.,\"ANSWER\": \"C\". If you think your previous thinking is incorrect, please start thinking completely from scratch."
MATH_retry = "Please reason step by step, and put your final answer within \\boxed{}. If you think your previous thinking is incorrect, please start thinking completely from scratch."

GPQA_temp = """{problem}
A) {A}
B) {B}
C) {C}
D) {D}

"""

GPQA_answer_prompt = " I should show my choice in the answer field with only the choice letter. </think> ANSWER:"
MATH_answer_prompt = "\n\nOh, I think I have found the final answer.\n\n**Final Answer** \\boxed{"
MATH_stop_think = "\n\nOh, I think I have finished thinking. </think>"

leap_prefix_MATH = """Please reason step by step, and when you get some intermediate results, please summarize them enlosed with <summarize> </summarize> and you will get the comments from peers. For example:

<summarize> In short, my current key insights about this problem are: Convert numbers to base 10 and set up the equations for the divisibility condition. Then simplify the equation and solve for \( b \). After that, find valid solutions, check for constraints, and sum them up for the final answer. And my current progress is: I have computed and confirmed the expressions for 
\\[
17_b = b + 7
\\]
and
\\[
97_b = 9b + 7.
\\]
I then set up the equation
\\[
9b + 7 = k(b + 7)
\\]
and derived the formula
\\[
b = \\frac{7(k - 1)}{9 - k}.
\\] </summarize> <comment> The comments from peers will be presented here. </comment>

After you get the final answer, return the final answer within \\boxed{}."""


leap_prefix_GPQA = """Please reason step by step, and when you get some intermediate results, please summarize them enlosed with <summarize> </summarize> and you will get the comments from peers. For example:

<summarize> In short, my current key insights about this problem are: Use the energy-time uncertainty principle ΔE * Δt ≈ ℏ to relate the lifetime τ of each quantum state to its energy uncertainty ΔE ≈ ℏ / τ. To distinguish two quantum states, the energy difference between them must be greater than the sum of their individual uncertainties: ΔE_diff > ΔE1 + ΔE2. And my current progress is: I calculated ΔE1 ≈ 6.58e-7 eV for τ1 = 1e-9 s and Δ21 ≈ 6.58e-8 eV for τ2 = 1e-8 s. Summing these gives the required minimum resolvable energy difference ≈ 7.218e-7 eV, so only option A is large enough to distinguish the states. </summarize> <comment> The comments from peers will be presented here. </comment>

After you get the final choice, show your choice in the answer field with only the choice letter, e.g.,\"ANSWER\": \"C\"."""


leap_subfix_MATH = """Okay, so I have this complex mathematical problem. And the user instruct that I should summarize what I've concluded with tags when I get some intermediate results. For example:

<summarize> In short, my current key insights about this problem are: Convert numbers to base 10 and set up the equations for the divisibility condition. Then simplify the equation and solve for \( b \). After that, find valid solutions, check for constraints, and sum them up for the final answer. And my current progress is: I have computed and confirmed the expressions for 
\\(
17_b = b + 7
\\)
and
\\(
97_b = 9b + 7.
\\)
I then set up the equation
\\(
9b + 7 = k(b + 7)
\\)
and derived the formula
\\(
b = \\frac{7(k - 1)}{9 - k}.
\\) </summarize>

Now, let's get back to the original problem."""


leap_subfix_GPQA = """Okay, so I have this complex problem. And the user instruct that I should summarize what I've concluded with tags when I get some intermediate results. For example:

<summarize> In short, my current key insights about this problem are: Use the energy-time uncertainty principle ΔE * Δt ≈ ℏ to relate the lifetime τ of each quantum state to its energy uncertainty ΔE ≈ ℏ / τ. To distinguish two quantum states, the energy difference between them must be greater than the sum of their individual uncertainties: ΔE_diff > ΔE1 + ΔE2. And my current progress is: I calculated ΔE1 ≈ 6.58e-7 eV for τ1 = 1e-9 s and Δ21 ≈ 6.58e-8 eV for τ2 = 1e-8 s. Summing these gives the required minimum resolvable energy difference ≈ 7.218e-7 eV, so only option A is large enough to distinguish the states. </summarize>

Now, let's get back to the original problem."""


leap_triggers = [
    "Alright, let's take a step back and summarize what we've figured out so far briefly.",
    "Wait, let me quickly recap what I've concluded so far.",
    "Alright, let me shortly review the conclusions I've drawn so I can move forward more efficiently.",
    "Hmm, a quick summary of what I've figured out might help streamline the next part of my reasoning.",
    "Hold on, I should summarize the key points briefly to ensure I'm on the right track.",
    "Okay, before continuing, let me put together a brief summary of the insights I've gathered so far.",
    "Okay, time to consolidate everything I've found into a concise summary."
]


summarize_triggers = [
    " <summarize> In short, my current conclusions are that",
    " <summarize> To summarize, based on my previous reasoning, I have currently found that",
    " <summarize> In conclusion, the current key takeaways and results are",
    " <summarize> In short, I've currently concluded that",
    " <summarize> To summarize, my recent findings are",
    " <summarize> In conclusion, the current insights and results I've gathered are",
]

comment_subfix = "Hmm, it seems that my peers have given me some comments, so let me check if anyone's conclusions are different from mine before I continue my own reasoning."


moa_template = """Problem: {problem}

You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:
"""

def get_leap():
    import random
    return random.choice(leap_triggers) + random.choice(summarize_triggers)