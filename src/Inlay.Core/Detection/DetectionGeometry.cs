using OpenCvSharp;

namespace Inlay.Detection;

internal static class DetectionGeometry
{
    internal static double CountEdges(Mat edges, Rect rect)
    {
        rect = ClampRect(rect, edges.Width, edges.Height);
        if (rect.Width <= 0 || rect.Height <= 0)
            return 0;
        using var region = new Mat(edges, rect);
        return Cv2.CountNonZero(region);
    }

    internal static Rect ScaleRect(Rect rect, double scale) => new(

    internal static Rect ClampRect(Rect rect, int width, int height)
    {
        var x = Math.Clamp(rect.X, 0, width - 1);
        var y = Math.Clamp(rect.Y, 0, height - 1);
        return new Rect(x, y, Math.Clamp(rect.Width, 1, width - x), Math.Clamp(rect.Height, 1, height - y));
    }

    internal static double IntersectionOverUnion(Rect a, Rect b)
    {
        var left = Math.Max(a.Left, b.Left);
        var top = Math.Max(a.Top, b.Top);
        var right = Math.Min(a.Right, b.Right);
        var bottom = Math.Min(a.Bottom, b.Bottom);
        var intersectionWidth = Math.Max(0, right - left);
        var intersectionHeight = Math.Max(0, bottom - top);
        var intersectionArea = (double)intersectionWidth * intersectionHeight;
        var unionArea = (double)a.Width * a.Height + (double)b.Width * b.Height - intersectionArea;
        return unionArea <= 0 ? 0 : intersectionArea / unionArea;
    }
}
