using Inlay.Detection;
using OpenCvSharp;
using Xunit;

namespace Inlay.Tests;

public sealed class ForegroundCardDetectionTests
{
    [Fact]
    public void RecognizesCenteredForegroundCardOverBlurredBackdropWithHighConfidence()
    {
        const int imageWidth = 1080;
        const int imageHeight = 2400;
        var foreground = new Rect(35, 699, 1011, 1001);

        using var backdrop = new Mat(imageHeight, imageWidth, MatType.CV_8UC3, new Scalar(18, 23, 25));
        var random = new Random(239);
        for (var index = 0; index < 90; index++)
        {
            var x = random.Next(0, imageWidth - 90);
            var y = random.Next(0, imageHeight - 90);
            var width = random.Next(24, 180);
            var height = random.Next(12, 100);
            var color = new Scalar(
                random.Next(15, 100),
                random.Next(15, 100),
                random.Next(15, 100));
            Cv2.Rectangle(backdrop, new Rect(x, y, width, height), color, -1);
        }

        using var blurredBackdrop = new Mat();
        Cv2.GaussianBlur(backdrop, blurredBackdrop, new Size(61, 61), 18);
        using var image = blurredBackdrop.Clone();

        using (var photo = new Mat(image, foreground))
        {
            for (var y = 0; y < foreground.Height; y += 20)
            {
                for (var x = 0; x < foreground.Width; x += 20)
                {
                    var color = new Scalar(
                        random.Next(35, 235),
                        random.Next(35, 235),
                        random.Next(35, 235));
                    Cv2.Rectangle(photo, new Rect(x, y, 12, 12), color, -1);
                }
            }

            Cv2.PutText(photo, "FOREGROUND IMAGE", new Point(70, 90),
                HersheyFonts.HersheySimplex, 1.5, new Scalar(245, 245, 245), 4);
            Cv2.Rectangle(photo, new Rect(0, 0, foreground.Width, foreground.Height),
                new Scalar(245, 245, 245), 3);
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.Equal(DetectionStatus.Extracted, result.Status);
        Assert.True(
            result.Best!.Confidence >= DetectionResult.AutomaticThreshold,
            $"Expected a recognized foreground card to exceed 90% confidence, got {result.Best.Confidence:P1}. " +
            $"Backdrop {result.Best.BackdropSeparation:P1}, boundary {result.Best.BoundaryContinuity:P1}, " +
            $"contrast {result.Best.BoundaryContrast:P1}, geometry {result.Best.GeometryFit:P1}.");
        Assert.True(result.Best.BackdropSeparation >= 0.85);
        Assert.True(result.Best.BoundaryContrast >= 0.80);
        Assert.InRange(result.Best.Bounds.X, foreground.X - 20, foreground.X + 20);
        Assert.InRange(result.Best.Bounds.Y, foreground.Y - 25, foreground.Y + 25);
        Assert.InRange(result.Best.Bounds.Right, foreground.Right - 20, foreground.Right + 20);
        Assert.InRange(result.Best.Bounds.Bottom, foreground.Bottom - 25, foreground.Bottom + 25);
    }
}
