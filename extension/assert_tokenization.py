import torch

def assert_same_tokenization(model_a, model_b, instruction: str) -> torch.Tensor:
    """
    Ensures both wrappers produce identical input_ids and attention_mask.
    Returns the shared (input_ids, attention_mask).
    """
    enc_a = model_a.tokenize_instructions_fn(instructions=[instruction])
    enc_b = model_b.tokenize_instructions_fn(instructions=[instruction])

    a_ids, a_mask = enc_a.input_ids.cpu(), enc_a.attention_mask.cpu()
    b_ids, b_mask = enc_b.input_ids.cpu(), enc_b.attention_mask.cpu()

    if not torch.equal(a_ids, b_ids) or not torch.equal(a_mask, b_mask):
        # Helpful diff
        mism = (a_ids != b_ids).nonzero(as_tuple=False)
        msg = (
            f"Tokenization mismatch for instruction: {instruction!r}\n"
            f"input_ids equal? {torch.equal(a_ids, b_ids)}\n"
            f"attention_mask equal? {torch.equal(a_mask, b_mask)}\n"
            f"First mismatched positions (up to 10): {mism[:10].tolist()}\n"
            f"a_ids[:50]={a_ids[0,:50].tolist()}\n"
            f"b_ids[:50]={b_ids[0,:50].tolist()}\n"
        )
        raise AssertionError(msg)

    return enc_a.input_ids, enc_a.attention_mask


# Unit test (doesn't require your models)
def _test_assert_same_tokenization_smoke():
    class DummyEnc: 
        def __init__(self, ids, mask): self.input_ids, self.attention_mask = ids, mask
    class DummyModel:
        def __init__(self, ids, mask): self._enc = DummyEnc(ids, mask)
        def tokenize_instructions_fn(self, instructions): return self._enc

    ids = torch.tensor([[1,2,3]])
    mask = torch.tensor([[1,1,1]])
    a = DummyModel(ids, mask)
    b = DummyModel(ids.clone(), mask.clone())
    out_ids, out_mask = assert_same_tokenization(a, b, "x")
    assert torch.equal(out_ids, ids) and torch.equal(out_mask, mask)

def test_yi_models():
    """Test tokenization consistency between yi-6b-chat and its uncensored version."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from pipeline.model_utils.model_factory import construct_model_base
    
    model_path = "01-ai/yi-6b-chat"
    model_path_uncensored = "spkgyk/Yi-6B-Chat-uncensored"
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    print(f"Loading uncensored model: {model_path_uncensored}")
    model_base_uncensored = construct_model_base(model_path_uncensored)
    
    # Test with a few sample instructions
    test_instructions = [
        "How do I make a cake?",
        "Tell me about the weather.",
        "Write a poem about nature.",
    ]
    
    print("\nTesting tokenization consistency...")
    for instruction in test_instructions:
        input_ids, attention_mask = assert_same_tokenization(
            model_base, model_base_uncensored, instruction
        )
        print(f"  ✓ '{instruction[:40]}...' - OK (len={input_ids.shape[1]})")
    
    print("\n✓ All tokenization tests passed!")


if __name__ == "__main__":
    _test_assert_same_tokenization_smoke()
    print("Smoke test ok\n")
    test_yi_models()
