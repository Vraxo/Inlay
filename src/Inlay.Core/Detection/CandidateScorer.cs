using OpenCvSharp;

namespace Inlay.Detection;

internal static class CandidateScorer
{
    internal static double MeasureRectangleGeometry(Mat edges, Rect rect)
    {
        var thickness = Math.Max(1, Math.Min(3, Math.Min(rect.Width, rect.Height) / 100));
        var top = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Y, rect.Width, thickness)) /
                  (rect.Width * (double)thickness);
        var bottom = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Bottom - thickness, rect.Width, thickness)) /
                     (rect.Width * (double)thickness);
        var left = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Y, thickness, rect.Height)) /
                   (rect.Height * (double)thickness);
        var right = DetectionGeometry.CountEdges(edges, new Rect(rect.Right - thickness, rect.Y, thickness, rect.Height)) /
                    (rect.Height * (double)thickness);

        // A good rectangle needs all four sides, not just a strong average
        // that can hide one missing or misplaced edge.
        return Math.Clamp(Math.Min(Math.Min(top, bottom), Math.Min(left, right)) * 3.0, 0, 1);
    }

    internal static double MeasureBoundaryContinuity(Mat edges, Rect rect)
    {
        var thickness = Math.Max(1, Math.Min(3, Math.Min(rect.Width, rect.Height) / 100));
        var top = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Y, rect.Width, thickness));
        var bottom = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Bottom - thickness, rect.Width, thickness));
        var left = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Y, thickness, rect.Height));
        var right = DetectionGeometry.CountEdges(edges, new Rect(rect.Right - thickness, rect.Y, thickness, rect.Height));
        var horizontal = (top + bottom) / (2.0 * rect.Width * thickness);
        var vertical = (left + right) / (2.0 * rect.Height * thickness);
        return Math.Clamp((horizontal + vertical) * 0.5 * 2.2, 0, 1);
    }

    internal static double MeasureEdgeStrength(Mat gray, Rect rect)
    {
        using var region = new Mat(gray, DetectionGeometry.ClampRect(rect, gray.Width, gray.Height));
        using var gradientX = new Mat();
        using var gradientY = new Mat();
        using var magnitude = new Mat();
        Cv2.Sobel(region, gradientX, MatType.CV_32F, 1, 0);
        Cv2.Sobel(region, gradientY, MatType.CV_32F, 0, 1);
        Cv2.Magnitude(gradientX, gradientY, magnitude);
        return Math.Clamp(Cv2.Mean(magnitude).Val0 / 90.0, 0, 1);
    }

    internal static double MeasureSharpnessContrast(Mat gray, Rect rect)
    {
        var inner = DetectionGeometry.ClampRect(new Rect(rect.X + rect.Width / 8, rect.Y + rect.Height / 8, rect.Width * 3 / 4, rect.Height * 3 / 4), gray.Width, gray.Height);
        var outer = DetectionGeometry.ClampRect(new Rect(Math.Max(0, rect.X - 12), Math.Max(0, rect.Y - 12),
            Math.Min(gray.Width - Math.Max(0, rect.X - 12), rect.Width + 24),
            Math.Min(gray.Height - Math.Max(0, rect.Y - 12), rect.Height + 24)), gray.Width, gray.Height);
        using var innerMat = new Mat(gray, inner);
        using var innerLap = new Mat();
        Cv2.Laplacian(innerMat, innerLap, MatType.CV_32F);
        Cv2.MeanStdDev(innerLap, out _, out var innerStd);
        var innerVariance = innerStd.Val0 * innerStd.Val0;

        using var outerMat = new Mat(gray, outer);
        using var outerLap = new Mat();
        Cv2.Laplacian(outerMat, outerLap, MatType.CV_32F);
        Cv2.MeanStdDev(outerLap, out _, out var outerStd);
        var outerVariance = outerStd.Val0 * outerStd.Val0;
        var contrast = (innerVariance - outerVariance) / Math.Max(1.0, innerVariance + outerVariance);
        return Math.Clamp(0.5 + contrast * 0.9, 0, 1);
    }
}
