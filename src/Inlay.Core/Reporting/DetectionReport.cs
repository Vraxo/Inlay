using Inlay.Detection;
using OpenCvSharp;

namespace Inlay.Reporting;

public sealed record DetectionReport(
    string Input,
    string Status,
    BoundsReport? Bounds,
    CornersReport Corners,
    double? Confidence,
    string Reason,
    string? Output);

public sealed record BoundsReport(int X, int Y, int Width, int Height);
public sealed record CornersReport(bool Detected, int Radius, bool MaskApplied);
