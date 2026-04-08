import 'dart:math' as math;
import 'dart:ui';

class TouchProcessorConfig {
  final double baseMultiplier;
  final double accelerationFactor;
  final double maxAcceleration;
  final double smoothingAlpha;
  final double predictionWeight;
  final int historySize;
  final double directionLockThreshold;
  final double minVelocity;
  final double maxVelocity;

  const TouchProcessorConfig({
    this.baseMultiplier = 1.0,
    this.accelerationFactor = 0.8,
    this.maxAcceleration = 3.0,
    this.smoothingAlpha = 0.4,
    this.predictionWeight = 0.25,
    this.historySize = 5,
    this.directionLockThreshold = 0.7,
    this.minVelocity = 0.0,
    this.maxVelocity = 5000.0,
  });

  static const TouchProcessorConfig defaultConfig = TouchProcessorConfig();
  
  static const TouchProcessorConfig precise = TouchProcessorConfig(
    baseMultiplier: 0.8,
    accelerationFactor: 0.5,
    maxAcceleration: 2.0,
    smoothingAlpha: 0.5,
    predictionWeight: 0.15,
  );

  static const TouchProcessorConfig fast = TouchProcessorConfig(
    baseMultiplier: 1.2,
    accelerationFactor: 1.0,
    maxAcceleration: 4.0,
    smoothingAlpha: 0.3,
    predictionWeight: 0.35,
  );
}

class TouchSample {
  final Offset position;
  final Offset delta;
  final DateTime timestamp;
  final double velocity;

  TouchSample({
    required this.position,
    required this.delta,
    required this.timestamp,
    required this.velocity,
  });
}

class TouchProcessor {
  final TouchProcessorConfig config;
  
  final List<TouchSample> _history = [];
  Offset _lastSmoothedDelta = Offset.zero;
  Offset _lastPosition = Offset.zero;
  DateTime? _lastTimestamp;
  double _currentVelocity = 0.0;
  Offset _lockedDirection = Offset.zero;
  bool _isDirectionLocked = false;
  
  int _skipCount = 0;
  static const int _maxSkipCount = 2;

  TouchProcessor({this.config = TouchProcessorConfig.defaultConfig});

  Offset process({
    required Offset rawDelta,
    required Offset position,
    required DateTime timestamp,
  }) {
    if (rawDelta.dx == 0 && rawDelta.dy == 0) {
      return Offset.zero;
    }

    final velocity = _calculateVelocity(rawDelta, timestamp);
    final acceleratedDelta = _applyAccelerationCurve(rawDelta, velocity);
    final smoothedDelta = _applySmoothing(acceleratedDelta);
    final lockedDelta = _applyDirectionLock(smoothedDelta);
    final predictedDelta = _applyPrediction(lockedDelta, position);

    _updateHistory(position, lockedDelta, timestamp, velocity);
    _lastSmoothedDelta = smoothedDelta;
    _lastPosition = position;
    _lastTimestamp = timestamp;
    _currentVelocity = velocity;

    return predictedDelta;
  }

  double _calculateVelocity(Offset delta, DateTime timestamp) {
    if (_lastTimestamp == null) {
      return 0.0;
    }

    final timeDelta = timestamp.difference(_lastTimestamp!).inMicroseconds;
    if (timeDelta <= 0) {
      return _currentVelocity;
    }

    final distance = delta.distance;
    final instantVelocity = distance / (timeDelta / 1000.0);

    final alpha = 0.3;
    final smoothedVelocity = _currentVelocity * (1 - alpha) + instantVelocity * alpha;

    return smoothedVelocity.clamp(config.minVelocity, config.maxVelocity);
  }

  Offset _applyAccelerationCurve(Offset delta, double velocity) {
    final normalizedVelocity = (velocity / config.maxVelocity).clamp(0.0, 1.0);
    
    final acceleration = 1.0 + normalizedVelocity * config.accelerationFactor;
    final clampedAcceleration = acceleration.clamp(1.0, config.maxAcceleration);
    
    final magnitude = delta.distance;
    if (magnitude == 0) return delta;

    final direction = delta / magnitude;
    final newMagnitude = magnitude * config.baseMultiplier * clampedAcceleration;
    
    return direction * newMagnitude;
  }

  Offset _applySmoothing(Offset delta) {
    final alpha = config.smoothingAlpha;
    final smoothed = _lastSmoothedDelta * (1 - alpha) + delta * alpha;
    
    final threshold = 0.5;
    if (smoothed.distance < threshold && delta.distance > threshold * 2) {
      _skipCount++;
      if (_skipCount <= _maxSkipCount) {
        return delta;
      }
    }
    _skipCount = 0;
    
    return smoothed;
  }

  Offset _applyDirectionLock(Offset delta) {
    if (delta.distance < 2.0) {
      return delta;
    }

    if (!_isDirectionLocked) {
      final absDx = delta.dx.abs();
      final absDy = delta.dy.abs();
      
      if (absDx > 0 && absDy > 0) {
        final ratio = absDx > absDy ? absDx / absDy : absDy / absDx;
        
        if (ratio > 1.0 / config.directionLockThreshold) {
          _isDirectionLocked = true;
          _lockedDirection = absDx > absDy 
              ? Offset(delta.dx > 0 ? 1 : -1, 0)
              : Offset(0, delta.dy > 0 ? 1 : -1);
        }
      }
    }

    if (_isDirectionLocked) {
      if (_lockedDirection.dx != 0) {
        return Offset(delta.dx, delta.dy * 0.1);
      } else {
        return Offset(delta.dx * 0.1, delta.dy);
      }
    }

    return delta;
  }

  Offset _applyPrediction(Offset delta, Offset position) {
    if (_history.length < 3) {
      return delta;
    }

    final recentHistory = _history.take(3).toList();
    
    Offset trend = Offset.zero;
    for (int i = 0; i < recentHistory.length - 1; i++) {
      trend += recentHistory[i].delta;
    }
    trend /= (recentHistory.length - 1);

    final predictedDelta = delta + trend * config.predictionWeight;
    
    final maxPredictedDistance = delta.distance * 1.5;
    if (predictedDelta.distance > maxPredictedDistance) {
      return delta;
    }

    return predictedDelta;
  }

  void _updateHistory(Offset position, Offset delta, DateTime timestamp, double velocity) {
    _history.insert(0, TouchSample(
      position: position,
      delta: delta,
      timestamp: timestamp,
      velocity: velocity,
    ));

    while (_history.length > config.historySize) {
      _history.removeLast();
    }
  }

  void reset() {
    _history.clear();
    _lastSmoothedDelta = Offset.zero;
    _lastPosition = Offset.zero;
    _lastTimestamp = null;
    _currentVelocity = 0.0;
    _lockedDirection = Offset.zero;
    _isDirectionLocked = false;
    _skipCount = 0;
  }

  void resetDirectionLock() {
    _isDirectionLocked = false;
    _lockedDirection = Offset.zero;
  }

  double get currentVelocity => _currentVelocity;
  
  bool get isDirectionLocked => _isDirectionLocked;
  
  Offset get lockedDirection => _lockedDirection;
  
  List<TouchSample> get history => List.unmodifiable(_history);
}

class TouchProcessorManager {
  static final TouchProcessorManager _instance = TouchProcessorManager._internal();
  factory TouchProcessorManager() => _instance;
  TouchProcessorManager._internal();

  final Map<String, TouchProcessor> _processors = {};
  TouchProcessorConfig _globalConfig = TouchProcessorConfig.defaultConfig;

  TouchProcessor getProcessor(String sessionId) {
    return _processors.putIfAbsent(
      sessionId,
      () => TouchProcessor(config: _globalConfig),
    );
  }

  void setGlobalConfig(TouchProcessorConfig config) {
    _globalConfig = config;
    for (final processor in _processors.values) {
      processor.reset();
    }
  }

  TouchProcessorConfig get globalConfig => _globalConfig;

  void removeProcessor(String sessionId) {
    _processors.remove(sessionId);
  }

  void resetAll() {
    for (final processor in _processors.values) {
      processor.reset();
    }
  }

  void dispose() {
    _processors.clear();
  }
}

class GestureEnhancer {
  static const double _tapTimeoutMs = 300;
  static const double _doubleTapTimeoutMs = 400;
  static const double _tapSlop = 50.0;
  static const double _longPressTimeoutMs = 600;

  DateTime? _lastTapTime;
  Offset? _lastTapPosition;
  int _tapCount = 0;

  bool isTap(Offset downPosition, Offset upPosition, DateTime downTime, DateTime upTime) {
    final duration = upTime.difference(downTime).inMilliseconds;
    final distance = (upPosition - downPosition).distance;
    
    return duration < _tapTimeoutMs && distance < _tapSlop;
  }

  bool isDoubleTap(Offset position, DateTime time) {
    if (_lastTapTime == null || _lastTapPosition == null) {
      _lastTapTime = time;
      _lastTapPosition = position;
      return false;
    }

    final timeDiff = time.difference(_lastTapTime!).inMilliseconds;
    final distance = (position - _lastTapPosition!).distance;

    if (timeDiff < _doubleTapTimeoutMs && distance < _tapSlop) {
      _lastTapTime = null;
      _lastTapPosition = null;
      return true;
    }

    _lastTapTime = time;
    _lastTapPosition = position;
    return false;
  }

  bool isLongPress(DateTime downTime, DateTime currentTime) {
    return currentTime.difference(downTime).inMilliseconds >= _longPressTimeoutMs;
  }

  void reset() {
    _lastTapTime = null;
    _lastTapPosition = null;
    _tapCount = 0;
  }
}

class EdgeSwipeDetector {
  static const double _edgeThreshold = 30.0;
  static const double _minSwipeDistance = 100.0;
  static const int _maxSwipeTimeMs = 500;

  Offset? _startPosition;
  DateTime? _startTime;
  Size? _screenSize;

  void start(Offset position, DateTime time, {Size? screenSize}) {
    _startPosition = position;
    _startTime = time;
    _screenSize = screenSize;
  }

  EdgeSwipeResult? end(Offset position, DateTime time) {
    if (_startPosition == null || _startTime == null) {
      return null;
    }

    final duration = time.difference(_startTime!).inMilliseconds;
    if (duration > _maxSwipeTimeMs) {
      return null;
    }

    final delta = position - _startPosition!;
    final distance = delta.distance;
    
    if (distance < _minSwipeDistance) {
      return null;
    }

    final isFromLeftEdge = _startPosition!.dx < _edgeThreshold;
    bool isFromRightEdge = false;
    bool isFromTopEdge = false;
    bool isFromBottomEdge = false;
    
    if (_screenSize != null) {
      isFromRightEdge = _startPosition!.dx > _screenSize!.width - _edgeThreshold;
      isFromTopEdge = _startPosition!.dy < _edgeThreshold;
      isFromBottomEdge = _startPosition!.dy > _screenSize!.height - _edgeThreshold;
    }

    EdgeSwipeDirection? direction;
    
    if (delta.dx.abs() > delta.dy.abs()) {
      direction = delta.dx > 0 ? EdgeSwipeDirection.right : EdgeSwipeDirection.left;
    } else {
      direction = delta.dy > 0 ? EdgeSwipeDirection.down : EdgeSwipeDirection.up;
    }

    if (isFromLeftEdge && direction == EdgeSwipeDirection.right) {
      return EdgeSwipeResult(direction: direction, fromEdge: EdgePosition.left);
    }
    if (isFromRightEdge && direction == EdgeSwipeDirection.left) {
      return EdgeSwipeResult(direction: direction, fromEdge: EdgePosition.right);
    }
    if (isFromTopEdge && direction == EdgeSwipeDirection.down) {
      return EdgeSwipeResult(direction: direction, fromEdge: EdgePosition.top);
    }
    if (isFromBottomEdge && direction == EdgeSwipeDirection.up) {
      return EdgeSwipeResult(direction: direction, fromEdge: EdgePosition.bottom);
    }

    return null;
  }

  void reset() {
    _startPosition = null;
    _startTime = null;
    _screenSize = null;
  }
}

enum EdgeSwipeDirection { left, right, up, down }
enum EdgePosition { left, right, top, bottom }

class EdgeSwipeResult {
  final EdgeSwipeDirection direction;
  final EdgePosition fromEdge;

  EdgeSwipeResult({required this.direction, required this.fromEdge});
}
