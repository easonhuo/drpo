from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


exp_path = Path("src/drpo/e8_multitask_exp_tuning.py")
text = exp_path.read_text(encoding="utf-8")
text = replace_once(text, "import queue\n", "", label="unused queue import")
text = replace_once(
    text,
    "    except Exception as exc:  # The plan records the exact fail-closed reason.\n",
    "    except Exception as exc:  # noqa: BLE001 - fail-closed reason capture\n",
    label="prepare recovery catch",
)
text = replace_once(
    text,
    "            _require_calibration_gate(config, output_root, base_model_path=base_model_path)\n"
    "            calibration_complete = True\n"
    "        except Exception as exc:\n",
    "            _require_calibration_gate(config, output_root, base_model_path=base_model_path)\n"
    "            calibration_complete = True\n"
    "        except Exception as exc:  # noqa: BLE001 - fail-closed reason capture\n",
    label="calibration recovery catch",
)
text = replace_once(
    text,
    "            _require_liveness_gate(config, output_root, base_model_path=base_model_path)\n"
    "            liveness_complete = True\n"
    "        except Exception as exc:\n",
    "            _require_liveness_gate(config, output_root, base_model_path=base_model_path)\n"
    "            liveness_complete = True\n"
    "        except Exception as exc:  # noqa: BLE001 - fail-closed reason capture\n",
    label="liveness recovery catch",
)
text = replace_once(
    text,
    "        def aggregate_group(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:\n"
    "            first = group[0]\n"
    "            return {\n"
    "                \"task\": task,\n",
    "        def aggregate_group(\n"
    "            group: Sequence[Mapping[str, Any]], task_name: str = task\n"
    "        ) -> dict[str, Any]:\n"
    "            first = group[0]\n"
    "            return {\n"
    "                \"task\": task_name,\n",
    label="legacy exponential aggregate closure",
)
exp_path.write_text(text, encoding="utf-8")

orchestration_path = Path("src/drpo/e8_multitask_orchestration.py")
text = orchestration_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "len(set(int(value) for value in gpu_ids))",
    "len({int(value) for value in gpu_ids})",
    label="gpu id set comprehension",
)
text = replace_once(
    text,
    "            except Exception as exc:  # pragma: no cover - caller-specific failures.\n",
    "            except Exception as exc:  # noqa: BLE001  # pragma: no cover\n",
    label="run-cell callback boundary",
)
text = replace_once(
    text,
    "                except Exception as exc:  # Keep failure evidence in scheduler output.\n",
    "                except Exception as exc:  # noqa: BLE001 - callback failure evidence\n",
    label="after-success callback boundary",
)
orchestration_path.write_text(text, encoding="utf-8")
