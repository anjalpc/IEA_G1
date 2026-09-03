from pathlib import Path


path = Path("/opt/unitree_mujoco/simulate/src/unitree_sdk2_bridge.h")
source = path.read_text(encoding="utf-8")

if 'card_overlay->arm_q(i, m.q())' in source:
    raise SystemExit(0)

source = source.replace(
    '#include "physics_joystick.h"',
    '#include "physics_joystick.h"\n#include "native_card_overlay.h"',
)
source = source.replace(
    'wireless_controller->joystick = joystick;\n    }',
    'wireless_controller->joystick = joystick;\n'
    '        card_overlay = std::make_unique<NativeCardOverlay>(model, data);\n'
    '    }',
)
source = source.replace(
    'm.kp() * (m.q() - mj_data_->sensordata[i]) +',
    'm.kp() * (card_overlay->arm_q(i, m.q()) - mj_data_->sensordata[i]) +',
)
source = source.replace(
    '        // lowstate\n',
    '        card_overlay->apply_hands();\n\n        // lowstate\n',
)
source = source.replace(
    '    std::unique_ptr<LowState_t> lowstate;\n',
    '    std::unique_ptr<LowState_t> lowstate;\n'
    '    std::unique_ptr<NativeCardOverlay> card_overlay;\n',
)

required = (
    '#include "native_card_overlay.h"',
    'card_overlay->arm_q(i, m.q())',
    'card_overlay->apply_hands()',
)
if not all(token in source for token in required):
    raise RuntimeError("native simulator source did not match expected revision")
path.write_text(source, encoding="utf-8")
