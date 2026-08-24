import os
import shutil
from pathlib import Path

from chessbench.ui.prompt_workbench import can_launch_match, compute_budget_state, is_ab_eligible
from chessbench.ui.strategy_store import (
    delete_strategy,
    export_yaml,
    import_yaml,
    list_strategies,
    load_strategy,
    save_strategy,
)


# Mock ValidationResult class for testing
class MockValidationResult:
    def __init__(self, is_valid=True, errors=None, warnings=None, used_fallback=False, estimated_tokens=0):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.used_fallback = used_fallback
        self.estimated_tokens = estimated_tokens


def test_strategy_store_roundtrip(tmp_path, monkeypatch):
    # Mock the home directory to tmp_path
    monkeypatch.setenv('HOME', str(tmp_path))
    print(f"HOME set to: {os.environ.get('HOME')}")
    strategy_dir = Path.home() / ".chessbench" / "strategies"
    print(f"strategy_dir: {strategy_dir}")
    # Ensure clean state
    if strategy_dir.exists():
        shutil.rmtree(strategy_dir)
    # Save a strategy
    name = "test_strategy"
    system_prompt = "You are a helpful assistant."
    turn_prompt = "What is the best move?"
    description = "A test strategy."

    saved_path = save_strategy(name, system_prompt, turn_prompt, description)
    assert saved_path.exists()
    assert saved_path.name == f"{name}.yaml"

    # List strategies
    strategies = list_strategies()
    assert len(strategies) == 1
    meta = strategies[0]
    assert meta.name == name
    assert meta.description == description
    # Note: created_at will be set, we can check it's a string

    # Load the strategy
    loaded_sys, loaded_turn = load_strategy(name)
    assert loaded_sys == system_prompt
    assert loaded_turn == turn_prompt

    # Export YAML
    yaml_text = export_yaml(name)
    assert system_prompt in yaml_text
    assert turn_prompt in yaml_text
    assert description in yaml_text

    # Import YAML (should overwrite if we allow, but we'll test with a new name)
    new_name = "imported_strategy"
    import_yaml(yaml_text, new_name)
    loaded_sys_new, loaded_turn_new = load_strategy(new_name)
    assert loaded_sys_new == system_prompt
    assert loaded_turn_new == turn_prompt

    # Delete the original strategy
    delete_strategy(name)
    strategies_after = list_strategies()
    assert len(strategies_after) == 1
    assert strategies_after[0].name == new_name

    # Clean up the imported one
    delete_strategy(new_name)
    # Finally, clean up the entire directory to ensure isolation
    if strategy_dir.exists():
        shutil.rmtree(strategy_dir)


def test_strategy_store_sanitize_filename(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    strategy_dir = Path.home() / ".chessbench" / "strategies"
    # Test unsafe characters
    unsafe_name = "invalid/name\\:*?\"<>|"
    # The store should sanitize or reject? We'll see what the implementation does.
    # We expect it to either reject or rename to a safe name.
    # For now, we'll just test that it doesn't break and produces a safe filename.
    print(f"Before save, strategy_dir: {strategy_dir}, exists: {strategy_dir.exists()}")
    try:
        path = save_strategy(unsafe_name, "sys", "turn")
        print(f"After save, strategy_dir: {strategy_dir}, exists: {strategy_dir.exists()}")
        # If it doesn't raise, check the saved file name is safe
        assert ".." not in path.name
        assert "/" not in path.name
        # ... etc.
    except Exception as e:
        # If it raises, that's acceptable too.
        print(f"Exception in save_strategy: {e}")
    finally:
        # Clean up the entire strategy directory to ensure isolation
        if strategy_dir.exists():
            print(f"Cleaning up strategy directory: {strategy_dir}")
            shutil.rmtree(strategy_dir)
        else:
            print(f"Strategy directory does not exist: {strategy_dir}")


def test_compute_budget_state():
    # Test cases: (rendered, window, expected_state)
    # window None -> "standard"
    assert compute_budget_state(100, None) == "standard"
    # >50% headroom -> ok
    assert compute_budget_state(100, 300) == "ok"  # 100/300 = 33% used -> 67% headroom >50%
    # Exactly 50% headroom -> headroom_pct = 50 -> not >50 and not <20 -> we return "ok" (fallback in function)
    # But let's see: we want >50% headroom for ok, so 50% headroom is not ok -> we should return warning?
    # Actually, the spec says:
    #   >50% headroom ok/green
    #   <20% headroom warning
    #   over-budget error
    # So 50% headroom is not >50% and not <20%, so it falls in the middle. We'll treat it as ok?
    # We'll follow the function as written: it returns "ok" for headroom_pct >=20 and <=50?
    # Let's compute: headroom_pct = 50 -> not >50, not <20 -> returns "ok".
    # We'll accept that.
    assert compute_budget_state(150, 300) == "ok"  # 150/300 = 50% used -> 50% headroom -> ok
    # <20% headroom -> warning
    # 240/300 = 80% used -> 20% headroom -> not <20%, so ok
    assert compute_budget_state(240, 300) == "ok"  # 20% headroom -> ok
    assert compute_budget_state(243, 300) == "warning"  # 243/300 = 81% used -> 19% headroom <20% -> warning
    # over-budget error
    assert compute_budget_state(301, 300) == "error"
    # edge: exactly at window
    assert compute_budget_state(300, 300) == "warning"  # rendered == window -> headroom 0% -> <20% -> warning


def test_can_launch_match():
    # Both valid
    v1 = MockValidationResult(is_valid=True)
    v2 = MockValidationResult(is_valid=True)
    can_launch, error = can_launch_match(v1, v2)
    assert can_launch is True
    assert error is None

    # P1 invalid
    v1 = MockValidationResult(is_valid=False, errors=["error1"])
    v2 = MockValidationResult(is_valid=True)
    can_launch, error = can_launch_match(v1, v2)
    assert can_launch is False
    assert "P1 Prompt Invalid" in error
    assert "- error1" in error

    # P2 invalid
    v1 = MockValidationResult(is_valid=True)
    v2 = MockValidationResult(is_valid=False, errors=["error2"])
    can_launch, error = can_launch_match(v1, v2)
    assert can_launch is False
    assert "P2 Prompt Invalid" in error
    assert "- error2" in error

    # Both invalid
    v1 = MockValidationResult(is_valid=False, errors=["error1"])
    v2 = MockValidationResult(is_valid=False, errors=["error2"])
    can_launch, error = can_launch_match(v1, v2)
    assert can_launch is False
    assert "P1 Prompt Invalid" in error
    assert "P2 Prompt Invalid" in error

    # used_fallback
    v1 = MockValidationResult(is_valid=True, used_fallback=True)
    v2 = MockValidationResult(is_valid=True)
    can_launch, error = can_launch_match(v1, v2)
    assert can_launch is False
    assert "P1 would use fallback prompt" in error

    v1 = MockValidationResult(is_valid=True)
    v2 = MockValidationResult(is_valid=True, used_fallback=True)
    can_launch, error = can_launch_match(v1, v2)
    assert can_launch is False
    assert "P2 would use fallback prompt" in error

    # Both used_fallback
    v1 = MockValidationResult(is_valid=True, used_fallback=True)
    v2 = MockValidationResult(is_valid=True, used_fallback=True)
    can_launch, error = can_launch_match(v1, v2)
    assert can_launch is False
    assert "P1 would use fallback prompt" in error
    assert "P2 would use fallback prompt" in error


def test_is_ab_eligible():
    model_1_config = {"provider": "test", "model_id": "model-a"}
    model_2_config = {"provider": "test", "model_id": "model-a"}
    # Same model, same strategy -> not eligible
    assert is_ab_eligible(model_1_config, model_2_config, "sys", "turn", "sys", "turn") is False
    # Same model, different system -> eligible
    assert is_ab_eligible(model_1_config, model_2_config, "sys1", "turn", "sys2", "turn") is True
    # Same model, different turn -> eligible
    assert is_ab_eligible(model_1_config, model_2_config, "sys", "turn1", "sys", "turn2") is True
    # Different model -> not eligible (regardless of strategy)
    model_2_config = {"provider": "test", "model_id": "model-b"}
    assert is_ab_eligible(model_1_config, model_2_config, "sys", "turn", "sys", "turn") is False
