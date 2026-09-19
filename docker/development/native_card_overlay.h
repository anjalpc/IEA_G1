#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

// Simulation-only left-card sequence. The Unitree RL policy retains control of
// the legs, waist, and right arm; only body actuators 15..21 are overlaid.
class NativeCardOverlay {
public:
    NativeCardOverlay(mjModel* model, mjData* data)
        : model_(model), data_(data), enabled_(std::getenv("IEA_G1_LEFT_CARD") != nullptr) {}

    double arm_q(int motor, double policy_q) {
        update();
        if (!enabled_ || motor < 15 || motor > 21 || phase_ == Phase::Waiting) {
            return policy_q;
        }
        const int joint = motor - 15;
        const double alpha = arm_alpha();
        return start_[joint] + alpha * (target_[joint] - start_[joint]);
    }

    void apply_hands() {
        update();
        if (model_->nu < 41) return;
        const double alpha = hand_alpha();
        for (int channel = 0; channel < 6; ++channel) {
            set_hand_actuator(29 + channel, right_neutral_[channel]);
            const double left = left_neutral_[channel] +
                alpha * (grip_[channel] - left_neutral_[channel]);
            set_hand_actuator(35 + channel, left);
        }
    }

private:
    enum class Phase { Waiting, Moving, Holding, Returning, Done };
    static constexpr double delay_ = 8.0;
    static constexpr double move_ = 5.0;
    static constexpr double hold_ = 5.0;
    static constexpr double hand_ramp_ = 1.5;

    static double smooth(double x) {
        x = std::max(0.0, std::min(1.0, x));
        return x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
    }

    void update() {
        if (!enabled_) return;
        const double elapsed = data_->time - delay_;
        Phase next = Phase::Waiting;
        if (elapsed >= 0.0 && elapsed < move_) next = Phase::Moving;
        else if (elapsed >= move_ && elapsed < move_ + hold_) next = Phase::Holding;
        else if (elapsed >= move_ + hold_ && elapsed < 2.0 * move_ + hold_) next = Phase::Returning;
        else if (elapsed >= 2.0 * move_ + hold_) next = Phase::Done;

        if (phase_ == Phase::Waiting && next == Phase::Moving) {
            for (int i = 0; i < 7; ++i) start_[i] = data_->sensordata[15 + i];
            std::cout << "LEFT CARD: moving from captured balanced pose" << std::endl;
        }
        if (next != phase_) {
            if (next == Phase::Holding) std::cout << "LEFT CARD: holding" << std::endl;
            if (next == Phase::Returning) std::cout << "LEFT CARD: returning" << std::endl;
            if (next == Phase::Done) std::cout << "LEFT CARD: complete" << std::endl;
        }
        phase_ = next;
    }

    double arm_alpha() const {
        const double elapsed = data_->time - delay_;
        if (phase_ == Phase::Moving) return smooth(elapsed / move_);
        if (phase_ == Phase::Holding) return 1.0;
        if (phase_ == Phase::Returning)
            return 1.0 - smooth((elapsed - move_ - hold_) / move_);
        return 0.0;
    }

    double hand_alpha() const {
        const double elapsed = data_->time - delay_;
        if (phase_ == Phase::Moving) return smooth(elapsed / hand_ramp_);
        if (phase_ == Phase::Holding) return 1.0;
        if (phase_ == Phase::Returning) {
            const double return_elapsed = elapsed - move_ - hold_;
            return 1.0 - smooth(return_elapsed / hand_ramp_);
        }
        return 0.0;
    }

    void set_hand_actuator(int actuator, double normalized) {
        const double low = model_->actuator_ctrlrange[2 * actuator];
        const double high = model_->actuator_ctrlrange[2 * actuator + 1];
        data_->ctrl[actuator] = low + normalized * (high - low);
    }

    mjModel* model_;
    mjData* data_;
    bool enabled_ = false;
    Phase phase_ = Phase::Waiting;
    std::array<double, 7> start_{};
    const std::array<double, 7> target_{
        -0.108721, 0.063594, -0.094819, 0.182400,
        -0.159618, -0.339249, 0.061755};
    const std::array<double, 6> grip_{0.780, 1.000, 0.400, 1.000, 1.000, 1.000};
    const std::array<double, 6> right_neutral_{
        0.399, 0.399, 0.050, 0.050, 0.049, 0.050};
    const std::array<double, 6> left_neutral_{
        0.400, 0.400, 0.049, 0.050, 0.049, 0.050};
};
