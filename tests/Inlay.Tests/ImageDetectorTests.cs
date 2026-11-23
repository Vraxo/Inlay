using Inlay.Detection;
using Inlay.Extraction;
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
        Assert.Contains("0 edge contours", result.Reason);
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
    public void PortraitCropRefinesOnlyTheTopEdgeAtSourceResolution()
    {
        using var image = new Mat(2400, 1080, MatType.CV_8UC3, new Scalar(85, 85, 85));
        var photo = new Rect(33, 693, 1008, 1014);
        Cv2.Rectangle(image, photo, new Scalar(125, 125, 125), -1);
        var random = new Random(23);
        for (var y = photo.Top + 24; y < photo.Bottom - 24; y += 18)
        {
            for (var x = photo.Left + 24; x < photo.Right - 24; x += 18)
            {
                var shade = (byte)random.Next(105, 150);
                Cv2.Rectangle(image, new Rect(x, y, 7, 7), new Scalar(shade, shade, shade), -1);
            }
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.Equal(693, result.Best!.Bounds.Y);
        Assert.Equal(1014, result.Best.Bounds.Height);
        Assert.Equal(1707, result.Best.Bounds.Bottom);
        Assert.Equal(0, result.Best.CornerRadius);
        Assert.InRange(result.Best.Bounds.X, 30, 36);
        Assert.InRange(result.Best.Bounds.Width, 1000, 1018);
        Assert.Contains(result.Status, new[] { DetectionStatus.Review, DetectionStatus.Extracted });
    }

    [Fact]
    public void DetectsRoundedCornersFromSourceResolutionEdges()
    {
        using var image = new Mat(2400, 1080, MatType.CV_8UC3, new Scalar(85, 85, 85));
        var photo = new Rect(33, 693, 1008, 1014);
        const int expectedRadius = 36;

        using var mask = ImageExtractor.CreateRoundedMask(photo.Width, photo.Height, expectedRadius);
        using (var photoRegion = new Mat(image, photo))
        {
            for (var row = 0; row < photo.Height; row++)
            {
                for (var column = 0; column < photo.Width; column++)
                {
                    var alpha = mask.At<byte>(row, column);
                    var value = (byte)((85 * (255 - alpha) + 125 * alpha + 127) / 255);
                    photoRegion.Set(row, column, new Vec3b(value, value, value));
                }
            }
        }

        var random = new Random(29);
        for (var y = photo.Top + 80; y < photo.Bottom - 80; y += 20)
        {
            for (var x = photo.Left + 80; x < photo.Right - 80; x += 20)
            {
                var shade = (byte)random.Next(105, 150);
                Cv2.Rectangle(image, new Rect(x, y, 7, 7), new Scalar(shade, shade, shade), -1);
            }
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.InRange(result.Best!.CornerRadius, expectedRadius - 5, expectedRadius + 5);
    }

    [Fact]
    public void MultiplePlausibleForegroundsRequireManualReview()
    {
        using var image = new Mat(900, 1300, MatType.CV_8UC3, new Scalar(85, 85, 85));
        var left = new Rect(100, 180, 420, 540);
        var right = new Rect(780, 180, 420, 540);
        DrawTexturedForeground(image, left, 47);
        DrawTexturedForeground(image, right, 47);

        var result = new ImageDetector().Detect(image);

        Assert.Equal(DetectionStatus.Review, result.Status);
        Assert.NotNull(result.Best);
        Assert.NotNull(result.Second);
    }

    [Fact]
    public void FindsSharpForegroundOverBlurredBackgroundWithoutVisibleBorder()
    {
        const int imageSize = 900;
        using var background = new Mat(imageSize, imageSize, MatType.CV_8UC3, new Scalar(90, 90, 90));
        var random = new Random(113);

        for (var y = 0; y < imageSize; y += 5)
        {
            for (var x = 0; x < imageSize; x += 5)
            {
                var shade = (byte)random.Next(55, 145);
                Cv2.Rectangle(background, new Rect(x, y, 3, 3), new Scalar(shade, shade, shade), -1);
            }
        }

        using var image = new Mat();
        Cv2.GaussianBlur(background, image, new Size(41, 41), 12);
        var foreground = new Rect(180, 190, 540, 520);
        random = new Random(127);

        for (var y = foreground.Top + 3; y < foreground.Bottom - 3; y += 7)
        {
            for (var x = foreground.Left + 3; x < foreground.Right - 3; x += 7)
            {
                var shade = (byte)random.Next(25, 230);
                Cv2.Rectangle(image, new Rect(x, y, 4, 4), new Scalar(shade, shade, shade), -1);
            }
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.InRange(result.Best!.Bounds.X, foreground.X - 35, foreground.X + 35);
        Assert.InRange(result.Best.Bounds.Y, foreground.Y - 35, foreground.Y + 35);
        Assert.InRange(result.Best.Bounds.Right, foreground.Right - 35, foreground.Right + 35);
        Assert.InRange(result.Best.Bounds.Bottom, foreground.Bottom - 35, foreground.Bottom + 35);
    }

    private static void DrawTexturedForeground(Mat image, Rect bounds, int seed)
    {
        Cv2.Rectangle(image, bounds, new Scalar(125, 125, 125), -1);
        Cv2.Rectangle(image, bounds, new Scalar(145, 145, 145), 3);

        var random = new Random(seed);
        for (var y = bounds.Top + 20; y < bounds.Bottom - 20; y += 16)
        {
            for (var x = bounds.Left + 20; x < bounds.Right - 20; x += 16)
            {
                var shade = (byte)random.Next(105, 150);
                Cv2.Rectangle(image, new Rect(x, y, 7, 7), new Scalar(shade, shade, shade), -1);
            }
        }
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
