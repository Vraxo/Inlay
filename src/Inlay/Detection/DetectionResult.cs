using OpenCvSharp;

namespace Inlay.Detection;

public enum DetectionStatus
{
    Extracted,
    Review,
    Rejected
}

public sealed record DetectionCandidate(
    Rect Bounds,
    double Confidence,
    double BoundaryContinuity,
    double EdgeStrength,
    double GeometryFit,
    double SharpnessContrast,
    double CornerFit,
    int CornerRadius,
    double AreaRatio);

public sealed record DetectionResult(
    DetectionStatus Status,
    DetectionCandidate? Best,
    DetectionCandidate? Second,
    string Reason)
{
    public const double AutomaticThreshold = 0.90;
    public const double ReviewThreshold = 0.65;
}
