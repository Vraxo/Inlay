using OpenCvSharp;

namespace Inlay.Detection;

internal static class FocusRegionDetector
{
    internal static void CreateFocusMap(Mat gray, Mat focusMap)
    {
        using var laplacian = new Mat();
        using var absolute = new Mat();
        Cv2.Laplacian(gray, laplacian, MatType.CV_32F, 3);
        Cv2.ConvertScaleAbs(laplacian, absolute);
        Cv2.GaussianBlur(absolute, focusMap, new Size(21, 21), 0);
    }

    internal static IReadOnlyList<Rect> FindCandidates(Mat focusMap)
    {
        using var mask = new Mat();
        using var closeKernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new Size(MakeOdd(focusMap.Width / 45), MakeOdd(focusMap.Height / 45)));
        using var openKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));

        Cv2.Threshold(focusMap, mask, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
        Cv2.MorphologyEx(mask, mask, MorphTypes.Close, closeKernel);
        Cv2.MorphologyEx(mask, mask, MorphTypes.Open, openKernel);
        Cv2.FindContours(mask, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        var candidates = new List<Rect>();
        foreach (var contour in contours)
        {
            var bounds = Cv2.BoundingRect(contour);
            var areaRatio = (double)bounds.Width * bounds.Height / (focusMap.Width * focusMap.Height);
            var aspectRatio = (double)bounds.Width / Math.Max(1, bounds.Height);

            if (bounds.Width < focusMap.Width * 0.18 ||
                bounds.Height < focusMap.Height * 0.12 ||
                areaRatio is < 0.04 or > 0.94 ||
                aspectRatio is < 0.25 or > 4.0)
                continue;

            var paddingX = Math.Max(4, bounds.Width / 40);
            var paddingY = Math.Max(4, bounds.Height / 40);
            var expanded = DetectionGeometry.ClampRect(
                new Rect(
                    bounds.X - paddingX,
                    bounds.Y - paddingY,
                    bounds.Width + paddingX * 2,
                    bounds.Height + paddingY * 2),
                focusMap.Width,
                focusMap.Height);

            if (candidates.All(existing => DetectionGeometry.IntersectionOverUnion(existing, expanded) < 0.82))
                candidates.Add(expanded);
        }

        return candidates
            .OrderByDescending(bounds => (long)bounds.Width * bounds.Height)
            .Take(24)
            .ToArray();
    }

    private static int MakeOdd(int value)
    {
        var size = Math.Max(7, value);
        return size % 2 == 0 ? size + 1 : size;
    }
}
