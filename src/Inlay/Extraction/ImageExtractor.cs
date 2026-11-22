using OpenCvSharp;
using Inlay.Detection;

namespace Inlay.Extraction;

public sealed class ImageExtractor
{
    public void Save(Mat source, Rect bounds, string outputPath, int radius, bool keepCorners)
    {
        if (bounds.X < 0 || bounds.Y < 0 || bounds.Width <= 0 || bounds.Height <= 0 ||
            bounds.Right > source.Width || bounds.Bottom > source.Height)
            throw new ArgumentOutOfRangeException(nameof(bounds), "Crop rectangle is outside the source image.");

        using var crop = new Mat(source, bounds);
        if (keepCorners || radius < 3)
        {
            Cv2.ImWrite(outputPath, crop);
            return;
        }

        using var bgra = new Mat();
        Cv2.CvtColor(crop, bgra, ColorConversionCodes.BGR2BGRA);
        using var mask = CreateRoundedMask(bounds.Width, bounds.Height, Math.Min(radius, Math.Min(bounds.Width, bounds.Height) / 2));
        Cv2.InsertChannel(mask, bgra, 3);
        Cv2.ImWrite(outputPath, bgra);
    }

    public static Mat CreateRoundedMask(int width, int height, int radius)
    {
        const int scale = 4;
        var scaledWidth = width * scale;
        var scaledHeight = height * scale;
        var scaledRadius = radius * scale;
        var mask = new Mat(scaledHeight, scaledWidth, MatType.CV_8UC1, Scalar.Black);

        if (radius <= 0)
        {
            mask.SetTo(Scalar.White);
        }
        else
        {
            if (scaledRadius * 2 < scaledWidth)
                Cv2.Rectangle(mask, new Point(scaledRadius, 0), new Point(scaledWidth - scaledRadius - 1, scaledHeight - 1), Scalar.White, -1);
            if (scaledRadius * 2 < scaledHeight)
                Cv2.Rectangle(mask, new Point(0, scaledRadius), new Point(scaledWidth - 1, scaledHeight - scaledRadius - 1), Scalar.White, -1);
            Cv2.Circle(mask, new Point(scaledRadius, scaledRadius), scaledRadius, Scalar.White, -1);
            Cv2.Circle(mask, new Point(scaledWidth - scaledRadius - 1, scaledRadius), scaledRadius, Scalar.White, -1);
            Cv2.Circle(mask, new Point(scaledRadius, scaledHeight - scaledRadius - 1), scaledRadius, Scalar.White, -1);
            Cv2.Circle(mask, new Point(scaledWidth - scaledRadius - 1, scaledHeight - scaledRadius - 1), scaledRadius, Scalar.White, -1);
        }

        var result = new Mat();
        Cv2.Resize(mask, result, new Size(width, height), interpolation: InterpolationFlags.Area);
        mask.Dispose();
        return result;
    }

    public void SavePreview(Mat source, DetectionResult result, string path)
    {
        using var preview = source.Clone();
        if (result.Best is not null)
        {
            var rect = result.Best.Bounds;
            Cv2.Rectangle(preview, rect, result.Status == DetectionStatus.Extracted ? new Scalar(50, 205, 50) : new Scalar(0, 165, 255), 3);
            Cv2.PutText(preview, $"{result.Status}  {result.Best.Confidence:P0}",
                new Point(Math.Max(8, rect.X), Math.Max(28, rect.Y - 10)),
                HersheyFonts.HersheySimplex, 0.8, new Scalar(0, 255, 255), 2);
        }
        if (result.Second is not null)
            Cv2.Rectangle(preview, result.Second.Bounds, Scalar.Red, 2);
        Cv2.ImWrite(path, preview);
    }
}
