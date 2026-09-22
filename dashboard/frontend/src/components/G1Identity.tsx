export function G1Identity() {
  return (
    <div className="g1-identity" aria-label="Unitree G1">
      <div className="g1-halo" />

      <svg
        className="g1-identity-svg"
        viewBox="0 0 240 280"
        role="img"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="shell" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#eef2f6" />
            <stop offset="52%" stopColor="#bcc6d0" />
            <stop offset="100%" stopColor="#7b8793" />
          </linearGradient>

          <linearGradient id="darkShell" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#242b33" />
            <stop offset="100%" stopColor="#0b0f14" />
          </linearGradient>

          <linearGradient id="visor" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#071017" />
            <stop offset="50%" stopColor="#0c2432" />
            <stop offset="100%" stopColor="#071017" />
          </linearGradient>
        </defs>

        <g className="g1-body">
          <path
            d="M78 42 Q120 12 162 42 L172 93 Q165 119 120 124 Q75 119 68 93 Z"
            fill="url(#darkShell)"
          />

          <path
            d="M82 52 Q120 31 158 52 L157 83 Q120 98 83 83 Z"
            fill="url(#visor)"
          />

          <rect
            x="95"
            y="67"
            width="50"
            height="5"
            rx="2.5"
            className="g1-eye"
          />

          <path
            d="M102 123 L138 123 L146 150 L94 150 Z"
            fill="#4c5965"
          />

          <path
            d="M76 151 Q120 132 164 151 L174 214 Q151 235 120 235 Q89 235 66 214 Z"
            fill="url(#shell)"
          />

          <path
            d="M91 158 Q120 147 149 158 L151 202 Q120 215 89 202 Z"
            fill="#252d35"
            opacity="0.78"
          />

          <path
            d="M69 161 L45 174 L38 223 L52 228 L72 188 Z"
            fill="url(#shell)"
          />

          <path
            d="M171 161 L195 174 L202 223 L188 228 L168 188 Z"
            fill="url(#shell)"
          />

          <path
            d="M93 229 L78 266 L101 269 L118 235 Z"
            fill="url(#shell)"
          />

          <path
            d="M147 229 L162 266 L139 269 L122 235 Z"
            fill="url(#shell)"
          />
        </g>
      </svg>

      <div className="g1-identity-copy">
        <strong>G1</strong>
        <span>HUMANOID PLATFORM</span>
      </div>
    </div>
  );
}
