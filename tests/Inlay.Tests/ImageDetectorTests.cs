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
