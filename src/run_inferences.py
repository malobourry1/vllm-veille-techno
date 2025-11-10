"""Script de test pour vLLM avec le modèle Granite."""

from vllm import LLM, SamplingParams


def run_inference():
    """Test basique de vLLM avec le modèle Granite."""
    llm = LLM(
        model="ibm-granite/granite-3.1-8b-instruct",
        max_model_len=2048,
        max_num_batched_tokens=2048,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

    prompts = [
        "Salut, peux-tu m'expliquer ce qu'est vLLM ?",
        "Écris-moi une fonction Python qui calcule la factorielle d'un nombre.",
        "Quels sont les avantages des modèles de langage open-source ?",
    ]
    print("\n🔤 Génération de réponses...")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")

        outputs = llm.generate(prompt, sampling_params)

        generated_text = outputs[0].outputs[0].text
        print(f"Réponse: {generated_text}")


if __name__ == "__main__":
    run_inference()
