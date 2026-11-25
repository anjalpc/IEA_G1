"""Causal time-weighted velocity window for controller prediction only."""
from collections import deque
import math

class VelocityWindow:
    def __init__(self, window_s=.6, maximum_gap_s=.25):
        if not all(math.isfinite(v) and v>0 for v in (window_s,maximum_gap_s)):
            raise ValueError('Window and maximum gap must be finite and positive')
        self.window_ns=round(window_s*1e9)
        self.gap_ns=round(maximum_gap_s*1e9)
        if self.window_ns<1 or self.gap_ns<1:
            raise ValueError('Window/gap below timestamp resolution')
        self.reset()

    def reset(self):
        self.samples=deque()
        self.started=None

    def add(self, stamp_ns, velocity):
        if (not isinstance(stamp_ns,int) or isinstance(stamp_ns,bool) or stamp_ns<0
                or len(velocity)!=3 or not all(math.isfinite(v) for v in velocity)):
            self.reset()
            raise ValueError('Invalid velocity sample')
        if self.samples and stamp_ns<=self.samples[-1][0]:
            self.reset()
            raise ValueError('Nonadvancing velocity timestamp')
        if self.samples and stamp_ns-self.samples[-1][0]>self.gap_ns:
            self.reset()
        if self.started is None:self.started=stamp_ns
        self.samples.append((stamp_ns,tuple(velocity)))
        cutoff=stamp_ns-self.window_ns
        # Retain one point before the cutoff for interpolation.
        while len(self.samples)>1 and self.samples[1][0]<=cutoff:
            self.samples.popleft()
        if stamp_ns-self.started<self.window_ns:return None
        values=list(self.samples)
        first_t,first_v=values[0]
        if first_t<cutoff:
            next_t,next_v=values[1]
            fraction=(cutoff-first_t)/(next_t-first_t)
            first_v=tuple(a+fraction*(b-a) for a,b in zip(first_v,next_v))
        values[0]=(cutoff,first_v)
        result=[0.,0.,0.]
        for (ta,va),(tb,vb) in zip(values,values[1:]):
            for axis in range(3):result[axis]+=(va[axis]*.5+vb[axis]*.5)*((tb-ta)/self.window_ns)
        if not all(math.isfinite(v) for v in result):
            self.reset()
            raise ValueError('Nonfinite averaged velocity')
        return tuple(result)

class PoseVelocityWindow:
    """Windowed planar pose displacement, expressed in the current child frame."""
    def __init__(self, window_s=.8, maximum_gap_s=.25):
        if not all(math.isfinite(v) and v>0 for v in (window_s,maximum_gap_s)):
            raise ValueError('Invalid pose window duration')
        self.window_ns=round(window_s*1e9)
        self.gap_ns=round(maximum_gap_s*1e9)
        if self.window_ns<1 or self.gap_ns<1:raise ValueError('Duration below timestamp resolution')
        self.reset()

    def reset(self):
        self.samples=deque()
        self.started=None
        self.previous_yaw=None
        self.unwrapped_yaw=None

    def add(self, stamp_ns, position, yaw):
        if (not isinstance(stamp_ns,int) or isinstance(stamp_ns,bool) or stamp_ns<0
                or len(position)!=2 or not all(math.isfinite(v) for v in (*position,yaw))):
            self.reset();raise ValueError('Invalid pose sample')
        if self.samples and stamp_ns<=self.samples[-1][0]:
            self.reset();raise ValueError('Nonadvancing pose timestamp')
        if self.samples and stamp_ns-self.samples[-1][0]>self.gap_ns:self.reset()
        if self.started is None:
            self.started=stamp_ns
            self.unwrapped_yaw=yaw
        else:
            delta=math.atan2(math.sin(yaw-self.previous_yaw),math.cos(yaw-self.previous_yaw))
            self.unwrapped_yaw+=delta
        self.previous_yaw=yaw
        self.samples.append((stamp_ns,(position[0],position[1],self.unwrapped_yaw)))
        cutoff=stamp_ns-self.window_ns
        while len(self.samples)>1 and self.samples[1][0]<=cutoff:self.samples.popleft()
        if stamp_ns-self.started<self.window_ns:return None
        earlier_t,earlier=self.samples[0]
        if earlier_t<cutoff:
            next_t,next_values=self.samples[1]
            fraction=(cutoff-earlier_t)/(next_t-earlier_t)
            earlier=tuple(a+fraction*(b-a) for a,b in zip(earlier,next_values))
        duration=self.window_ns/1e9
        dx=(position[0]-earlier[0])/duration
        dy=(position[1]-earlier[1])/duration
        wz=(self.unwrapped_yaw-earlier[2])/duration
        velocity=(math.cos(yaw)*dx+math.sin(yaw)*dy,
                  -math.sin(yaw)*dx+math.cos(yaw)*dy,wz)
        if not all(math.isfinite(v) for v in velocity):
            self.reset();raise ValueError('Nonfinite pose-derived velocity')
        return velocity
