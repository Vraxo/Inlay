using Inlay.Detection;
using OpenCvSharp;
using Xunit;

namespace Inlay.Tests;

public sealed class ImageDetectorTests
{
    [Fact]
    public void FlatScreenshotExplainsWhyNoCandidateWasFound()
    {
        using var image = new Mat(400, 300, MatType.CV_8UC3, new Scalar(90, 90, 90));

        var result = new ImageDetector().Detect(image);

        Assert.Equal(DetectionStatus.Rejected, result.Status);
        Assert.Null(result.Best);
        Assert.Contains("0 contours examined", result.Reason);
    }


    [Fact]
    public void HorizontalBoundsPreferFullImageWhenAnInternalVerticalEdgeIsStronger()
    {
        using var image = new Mat(1800, 1080, MatType.CV_8UC3, new Scalar(85, 85, 85));
        var photo = new Rect(32, 300, 1016, 1200);
        Cv2.Rectangle(image, photo, new Scalar(125, 125, 125), -1);
        Cv2.Rectangle(image, photo, new Scalar(145, 145, 145), 3);

        // A strong internal vertical edge can be mistaken for the image's left side.
        Cv2.Line(image, new Point(378, photo.Top + 4), new Point(378, photo.Bottom - 4), new Scalar(225, 225, 225), 5);

        var random = new Random(17);
        for (var y = photo.Top + 20; y < photo.Bottom - 20; y += 12)
        {
            for (var x = photo.Left + 20; x < photo.Right - 20; x += 12)
            {
                var shade = (byte)random.Next(105, 150);
                Cv2.Rectangle(image, new Rect(x, y, 5, 5), new Scalar(shade, shade, shade), -1);
            }
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.InRange(result.Best!.Bounds.X, 20, 50);
        Assert.InRange(result.Best.Bounds.Y, 280, 320);
        Assert.InRange(result.Best.Bounds.Width, 990, 1040);
        Assert.InRange(result.Best.Bounds.Height, 1170, 1230);
        Assert.NotEqual(DetectionStatus.Rejected, result.Status);
        Assert.DoesNotContain("%", result.Reason);
    }

    [Fact]
    public void SofterEdgesCanRecoverALowContrastForegroundBoundary()
    {
        using var image = new Mat(900, 900, MatType.CV_8UC3, new Scalar(100, 100, 100));
        Cv2.Rectangle(image, new Rect(180, 180, 540, 540), new Scalar(125, 125, 125), -1);

        for (var y = 198; y < 700; y += 48)
        {
            for (var x = 198; x < 700; x += 48)
            {
                var shade = (x / 48 + y / 48) % 2 == 0 ? 142 : 112;
                Cv2.Rectangle(image, new Rect(x, y, 24, 24), new Scalar(shade, shade, shade), -1);
            }
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.InRange(result.Best!.Bounds.X, 165, 195);
        Assert.InRange(result.Best.Bounds.Y, 165, 195);
        Assert.InRange(result.Best.Bounds.Width, 510, 570);
        Assert.InRange(result.Best.Bounds.Height, 510, 570);
    }
}
