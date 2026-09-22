export type ActionGroup =
  | "posture"
  | "locomotion"
  | "interaction"
  | "performance"
  | "custom";

export type ActionSource =
  | "loco"
  | "arm"
  | "remote"
  | "mimic"
  | "custom";

export interface G1Action {
  id: string;
  label: string;
  group: ActionGroup;
  source: ActionSource;

  /**
   * "direct":
   *   Current public SDK exposes a direct API.
   *
   * "discover":
   *   Availability should be confirmed from the connected robot,
   *   GetActionList(), GetMimicMotion(), firmware capability data,
   *   or our backend configuration.
   */
  availability: "direct" | "discover";

  dangerous?: boolean;
}

export const G1_ACTIONS: G1Action[] = [
  // ---------------------------------------------------------
  // Posture / safety
  // ---------------------------------------------------------
    {
    id: "damping",
    label: "Damping",
    group: "posture",
    source: "loco",
    availability: "direct",
  },
  {
    id: "zero_torque",
    label: "Zero Torque",
    group: "posture",
    source: "loco",
    availability: "direct",
    dangerous: true,
  },
  {
    id: "stand_up",
    label: "Stand Up",
    group: "posture",
    source: "loco",
    availability: "direct",
  },
  {
    id: "sit",
    label: "Seated Mode",
    group: "posture",
    source: "loco",
    availability: "direct",
  },
  {
    id: "squat",
    label: "Squat",
    group: "posture",
    source: "loco",
    availability: "direct",
  },
  {
    id: "high_stand",
    label: "High Stand",
    group: "posture",
    source: "loco",
    availability: "direct",
  },
  {
    id: "low_stand",
    label: "Low Stand",
    group: "posture",
    source: "loco",
    availability: "direct",
  },
  {
    id: "balance_stand",
    label: "Balance Stand",
    group: "posture",
    source: "loco",
    availability: "direct",
  },
  {
    id: "locked_standing",
    label: "Locked Standing",
    group: "posture",
    source: "remote",
    availability: "discover",
  },
  {
    id: "lie_to_stand",
    label: "Lie → Stand",
    group: "posture",
    source: "remote",
    availability: "discover",
  },

  // ---------------------------------------------------------
  // Locomotion
  // ---------------------------------------------------------
  {
    id: "stop_move",
    label: "Stop Movement",
    group: "locomotion",
    source: "loco",
    availability: "direct",
  },
  {
    id: "regular_mode",
    label: "Regular Mode",
    group: "locomotion",
    source: "remote",
    availability: "discover",
  },
  {
    id: "running_mode",
    label: "Running Mode",
    group: "locomotion",
    source: "remote",
    availability: "discover",
  },
  {
    id: "climb_mode",
    label: "Climb Mode",
    group: "locomotion",
    source: "remote",
    availability: "discover",
  },
  {
    id: "low_speed",
    label: "Low Speed",
    group: "locomotion",
    source: "loco",
    availability: "direct",
  },
  {
    id: "high_speed",
    label: "High Speed",
    group: "locomotion",
    source: "loco",
    availability: "direct",
  },
  {
    id: "continuous_gait",
    label: "Continuous Gait",
    group: "locomotion",
    source: "loco",
    availability: "direct",
  },

  // ---------------------------------------------------------
  // Interactions
  // ---------------------------------------------------------
  {
    id: "shake_hand",
    label: "Handshake",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "wave",
    label: "Wave Hand",
    group: "interaction",
    source: "loco",
    availability: "direct",
  },
  {
    id: "wave_turn",
    label: "Wave + Turn",
    group: "interaction",
    source: "loco",
    availability: "direct",
  },
  {
    id: "reject",
    label: "Reject",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "face_wave",
    label: "Face Wave",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "high_wave",
    label: "High Wave",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "clap",
    label: "Clap",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "high_five",
    label: "High Five",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "hug",
    label: "Hug",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "hands_up",
    label: "Hands Up",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "right_hand_up",
    label: "Right Hand Up",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "heart",
    label: "Heart",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "right_heart",
    label: "Right Heart",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "x_ray",
    label: "X-Ray",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "left_kiss",
    label: "Left Kiss",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "right_kiss",
    label: "Right Kiss",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "two_hand_kiss",
    label: "Two-Hand Kiss",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },
  {
    id: "release_arm",
    label: "Release Arm",
    group: "interaction",
    source: "arm",
    availability: "direct",
  },

  // Remote-guide actions that need capability confirmation.
  {
    id: "flex_left",
    label: "Flex Left",
    group: "interaction",
    source: "remote",
    availability: "discover",
  },
  {
    id: "flex_right",
    label: "Flex Right",
    group: "interaction",
    source: "remote",
    availability: "discover",
  },
  {
    id: "flex_both",
    label: "Flex Both",
    group: "interaction",
    source: "remote",
    availability: "discover",
  },
  {
    id: "stepping_standing",
    label: "Stepping / Standing",
    group: "interaction",
    source: "remote",
    availability: "discover",
  },

  
  // ---------------------------------------------------------
  // Performance / mimic
  // ---------------------------------------------------------
  {
    id: "dance_mode",
    label: "Dance Mode",
    group: "performance",
    source: "remote",
    availability: "discover",
  },
  {
    id: "dance_1",
    label: "Dance 1",
    group: "performance",
    source: "mimic",
    availability: "discover",
  },
  {
    id: "dance_2",
    label: "Dance 2",
    group: "performance",
    source: "mimic",
    availability: "discover",
  },
  {
    id: "dance_3",
    label: "Dance 3",
    group: "performance",
    source: "mimic",
    availability: "discover",
  },
  {
    id: "kung_fu",
    label: "Kung Fu",
    group: "performance",
    source: "mimic",
    availability: "discover",
  },
  {
    id: "roll_up",
    label: "Roll Up",
    group: "performance",
    source: "mimic",
    availability: "discover",
  },
  {
    id: "jeet_kune_do",
    label: "Jeet Kune Do",
    group: "performance",
    source: "mimic",
    availability: "discover",
  },
  {
    id: "twist",
    label: "Twist",
    group: "performance",
    source: "mimic",
    availability: "discover",
  },
];
