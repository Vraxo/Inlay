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
        Assert.NotEqual(DetectionStatus.Rejected, result.Status);
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
    public void CenteredForegroundReceivesAHigherScoreThanAnOtherwiseIdenticalOffsetForeground()
    {
        using var centeredImage = new Mat(900, 1400, MatType.CV_8UC3, new Scalar(85, 85, 85));
        using var offsetImage = new Mat(900, 1400, MatType.CV_8UC3, new Scalar(85, 85, 85));
        DrawTexturedForeground(centeredImage, new Rect(480, 200, 440, 500), 71);
        DrawTexturedForeground(offsetImage, new Rect(900, 200, 440, 500), 71);

        var detector = new ImageDetector();
        var centeredResult = detector.Detect(centeredImage);
        var offsetResult = detector.Detect(offsetImage);

        Assert.NotNull(centeredResult.Best);
        Assert.NotNull(offsetResult.Best);
        Assert.True(
            centeredResult.Best!.Confidence > offsetResult.Best!.Confidence,
            $"Expected centered score {centeredResult.Best.Confidence:F3} to exceed offset score {offsetResult.Best.Confidence:F3}.");
    }

    [Fact]
    public void FindsWideForegroundAtPhoneResolution()
    {
        using var image = new Mat(2400, 1080, MatType.CV_8UC3, new Scalar(85, 85, 85));
        var foreground = new Rect(32, 857, 1016, 683);
        DrawTexturedForeground(image, foreground, 83);

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.NotEqual(DetectionStatus.Rejected, result.Status);
        Assert.InRange(result.Best!.Bounds.X, foreground.X - 2, foreground.X + 2);
        Assert.InRange(result.Best.Bounds.Y, foreground.Y - 12, foreground.Y + 12);
        Assert.InRange(result.Best.Bounds.Right, foreground.Right - 2, foreground.Right + 2);
        Assert.InRange(result.Best.Bounds.Bottom, foreground.Bottom - 12, foreground.Bottom + 12);
    }

    [Fact]
    public void FindsWideCenteredImageAgainstBlurredPhoneBackdrop()
    {
        const int imageWidth = 921;
        const int imageHeight = 2048;
        var foreground = new Rect(25, 727, 870, 587);

        using var background = new Mat(imageHeight, imageWidth, MatType.CV_8UC3, new Scalar(18, 23, 25));
        var random = new Random(43);

        for (var index = 0; index < 80; index++)
        {
            var x = random.Next(0, imageWidth - 90);
            var y = random.Next(0, imageHeight - 90);
            var width = random.Next(24, 180);
            var height = random.Next(12, 100);
            var color = new Scalar(random.Next(15, 100), random.Next(15, 100), random.Next(15, 100));
            Cv2.Rectangle(background, new Rect(x, y, width, height), color, -1);
        }

        using var blurredBackground = new Mat();
        Cv2.GaussianBlur(background, blurredBackground, new Size(61, 61), 18);
        using var image = blurredBackground.Clone();

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

            Cv2.PutText(photo, "ACTUALLY SHUT UP", new Point(35, 72),
                HersheyFonts.HersheySimplex, 1.55, new Scalar(245, 245, 245), 4);
            Cv2.PutText(photo, "FOREGROUND IMAGE", new Point(90, 490),
                HersheyFonts.HersheySimplex, 1.15, new Scalar(235, 235, 235), 3);
            Cv2.Line(photo, new Point(0, 0), new Point(foreground.Width - 1, 0),
                new Scalar(230, 230, 230), 2);
            Cv2.Line(photo, new Point(foreground.Width - 1, 0),
                new Point(foreground.Width - 1, foreground.Height - 1), new Scalar(245, 245, 245), 5);
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.NotEqual(DetectionStatus.Rejected, result.Status);
        Assert.InRange(result.Best!.Bounds.X, foreground.X - 30, foreground.X + 30);
        Assert.InRange(result.Best.Bounds.Y, foreground.Y - 30, foreground.Y + 30);
        Assert.InRange(result.Best.Bounds.Right, foreground.Right - 30, foreground.Right + 30);
        Assert.InRange(result.Best.Bounds.Bottom, foreground.Bottom - 30, foreground.Bottom + 30);
        Assert.True(result.Best.Bounds.Width > result.Best.Bounds.Height);
        Assert.Equal(0, result.Best.CornerRadius);
    }

    [Fact]
    public void FindsMostlyCenteredPortraitImageAgainstUniformlyBlurredBackground()
    {
        using var background = new Mat(1920, 1080, MatType.CV_8UC3);
        Cv2.Randu(background, new Scalar(0, 0, 0), new Scalar(256, 256, 256));
        using var image = new Mat();
        Cv2.GaussianBlur(background, image, new Size(61, 61), 16);

        var foreground = new Rect(250, 500, 580, 850);
        using (var region = new Mat(image, foreground))
            Cv2.Randu(region, new Scalar(0, 0, 0), new Scalar(256, 256, 256));

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.NotEqual(DetectionStatus.Rejected, result.Status);
        Assert.InRange(result.Best!.Bounds.X, foreground.X - 55, foreground.X + 55);
        Assert.InRange(result.Best.Bounds.Y, foreground.Y - 55, foreground.Y + 55);
        Assert.InRange(result.Best.Bounds.Width, foreground.Width - 65, foreground.Width + 65);
        Assert.InRange(result.Best.Bounds.Height, foreground.Height - 65, foreground.Height + 65);
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

    [Fact]
    public void KeepsTheKnownWidePhoneCropGeometry()
    {
        const int imageWidth = 1080;
        const int imageHeight = 2400;
        var foreground = new Rect(30, 840, 1020, 720);

        using var background = new Mat(imageHeight, imageWidth, MatType.CV_8UC3, new Scalar(18, 23, 25));
        var random = new Random(149);
        for (var index = 0; index < 80; index++)
        {
            var x = random.Next(0, imageWidth - 90);
            var y = random.Next(0, imageHeight - 90);
            var width = random.Next(24, 180);
            var height = random.Next(12, 100);
            var color = new Scalar(random.Next(15, 100), random.Next(15, 100), random.Next(15, 100));
            Cv2.Rectangle(background, new Rect(x, y, width, height), color, -1);
        }

        using var blurredBackground = new Mat();
        Cv2.GaussianBlur(background, blurredBackground, new Size(61, 61), 18);
        using var image = blurredBackground.Clone();

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

            Cv2.PutText(photo, "COMPLETE FOREGROUND IMAGE", new Point(55, 75),
                HersheyFonts.HersheySimplex, 1.35, new Scalar(245, 245, 245), 4);
            Cv2.Line(photo, new Point(0, 0), new Point(foreground.Width - 1, 0),
                new Scalar(230, 230, 230), 2);
            Cv2.Line(photo, new Point(foreground.Width - 1, 0),
                new Point(foreground.Width - 1, foreground.Height - 1), new Scalar(245, 245, 245), 5);
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.NotEqual(DetectionStatus.Rejected, result.Status);
        Assert.InRange(result.Best!.Bounds.X, foreground.X - 12, foreground.X + 12);
        Assert.InRange(result.Best.Bounds.Y, foreground.Y - 16, foreground.Y + 16);
        Assert.InRange(result.Best.Bounds.Right, foreground.Right - 12, foreground.Right + 12);
        Assert.InRange(result.Best.Bounds.Bottom, foreground.Bottom - 16, foreground.Bottom + 16);
        Assert.Equal(0, result.Best.CornerRadius);
    }

    [Fact]
    public void DoesNotCropIntoAFullImageAroundItsSharperCentralSubject()
    {
        const int imageWidth = 1080;
        const int imageHeight = 1920;
        var foreground = new Rect(90, 420, 900, 1080);

        using var backdrop = new Mat(imageHeight, imageWidth, MatType.CV_8UC3);
        Cv2.Randu(backdrop, new Scalar(10, 10, 10), new Scalar(90, 90, 90));
        using var blurredBackdrop = new Mat();
        Cv2.GaussianBlur(backdrop, blurredBackdrop, new Size(61, 61), 18);
        using var image = blurredBackdrop.Clone();

        var random = new Random(157);
        using (var photo = new Mat(image, foreground))
        {
            // The complete photograph is clearly sharper than its blurred backdrop,
            // but its broad areas are less detailed than the subject in the middle.
            for (var y = 0; y < foreground.Height; y += 10)
            {
                for (var x = 0; x < foreground.Width; x += 10)
                {
                    var shade = random.Next(75, 180);
                    Cv2.Rectangle(photo, new Rect(x, y, 6, 6),
                        new Scalar(shade, shade, shade), -1);
                }
            }

            // A roughly circular, highly detailed subject deliberately has no
            // rectangular boundary for the detector to mistake for the photo edge.
            var centerX = foreground.Width / 2;
            var centerY = foreground.Height / 2;
            const int radius = 245;
            for (var y = centerY - radius; y <= centerY + radius; y += 3)
            {
                for (var x = centerX - radius; x <= centerX + radius; x += 3)
                {
                    var dx = x - centerX;
                    var dy = y - centerY;
                    if (dx * dx + dy * dy > radius * radius)
                        continue;

                    var shade = random.Next(0, 256);
                    Cv2.Rectangle(photo, new Rect(x, y, 2, 2),
                        new Scalar(shade, shade, shade), -1);
                }
            }
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.NotEqual(DetectionStatus.Rejected, result.Status);
        Assert.InRange(result.Best!.Bounds.X, foreground.X - 25, foreground.X + 25);
        Assert.InRange(result.Best.Bounds.Y, foreground.Y - 25, foreground.Y + 25);
        Assert.InRange(result.Best.Bounds.Right, foreground.Right - 25, foreground.Right + 25);
        Assert.InRange(result.Best.Bounds.Bottom, foreground.Bottom - 25, foreground.Bottom + 25);
    }

    [Fact]
    public void PrefersTheCompleteImageOverASharperRectangularRegionInsideIt()
    {
        const int imageWidth = 1080;
        const int imageHeight = 2400;
        var foreground = new Rect(29, 729, 1022, 942);
        var detailedRegion = new Rect(360, 900, 375, 600);

        using var background = new Mat(imageHeight, imageWidth, MatType.CV_8UC3, new Scalar(18, 23, 25));
        var random = new Random(163);
        for (var index = 0; index < 100; index++)
        {
            var x = random.Next(0, imageWidth - 90);
            var y = random.Next(0, imageHeight - 90);
            var width = random.Next(24, 180);
            var height = random.Next(12, 100);
            var color = new Scalar(random.Next(15, 100), random.Next(15, 100), random.Next(15, 100));
            Cv2.Rectangle(background, new Rect(x, y, width, height), color, -1);
        }

        using var blurredBackground = new Mat();
        Cv2.GaussianBlur(background, blurredBackground, new Size(61, 61), 18);
        using var image = blurredBackground.Clone();

        using (var photo = new Mat(image, foreground))
        {
            // The full photograph contains smooth, visible color variation.
            // It is low-detail, but it is not a blurred extension of the backdrop.
            for (var y = 0; y < foreground.Height; y++)
            {
                for (var x = 0; x < foreground.Width; x++)
                {
                    var wave = 18 * Math.Sin(x / 92.0) + 13 * Math.Cos(y / 81.0) +
                               8 * Math.Sin((x + y) / 143.0);
                    var baseValue = Math.Clamp(112 + wave, 0, 255);
                    photo.Set(y, x, new Vec3b(
                        (byte)Math.Clamp(baseValue - 18, 0, 255),
                        (byte)Math.Clamp(baseValue + 4, 0, 255),
                        (byte)Math.Clamp(baseValue + 20, 0, 255)));
                }
            }

            // A larger, moderately detailed visual element creates a plausible
            // competing crop around the sharper central subject.
            var mediumRegion = new Rect(255, 210, 602, 572);
            for (var y = mediumRegion.Y; y < mediumRegion.Bottom; y += 14)
            {
                for (var x = mediumRegion.X; x < mediumRegion.Right; x += 14)
                {
                    var shade = random.Next(65, 180);
                    Cv2.Rectangle(photo, new Rect(x, y, 8, 8),
                        new Scalar(shade, shade + 8, shade + 16), -1);
                }
            }

            // This is the sharp inner region reported by Inlay's diagnostics.
            // Its boundary is an object/detail boundary, not the photo boundary.
            var inner = new Rect(
                detailedRegion.X - foreground.X,
                detailedRegion.Y - foreground.Y,
                detailedRegion.Width,
                detailedRegion.Height);
            for (var y = inner.Y; y < inner.Bottom; y += 4)
            {
                for (var x = inner.X; x < inner.Right; x += 4)
                {
                    var shade = random.Next(0, 256);
                    Cv2.Rectangle(photo, new Rect(x, y, 2, 2),
                        new Scalar(shade, 255 - shade, (shade * 3) % 256), -1);
                }
            }
        }

        var result = new ImageDetector().Detect(image);

        Assert.NotNull(result.Best);
        Assert.NotEqual(DetectionStatus.Rejected, result.Status);
        Assert.InRange(result.Best!.Bounds.X, foreground.X - 25, foreground.X + 25);
        Assert.InRange(result.Best.Bounds.Y, foreground.Y - 25, foreground.Y + 25);
        Assert.InRange(result.Best.Bounds.Right, foreground.Right - 25, foreground.Right + 25);
        Assert.InRange(result.Best.Bounds.Bottom, foreground.Bottom - 25, foreground.Bottom + 25);
    }
}
