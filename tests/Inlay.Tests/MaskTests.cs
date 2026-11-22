using Xunit;

using Inlay.Extraction;
using OpenCvSharp;

namespace Inlay.Tests;

public sealed class MaskTests
{
    [Fact]
    public void SquareMaskIsFullyOpaque()
    {
        using var mask = ImageExtractor.CreateRoundedMask(40, 30, 0);
        Assert.Equal((byte)255, mask.At<byte>(0, 0));
        Assert.Equal(255, mask.At<byte>(15, 20));
    }

    [Fact]
    public void RoundedMaskPreservesCenterAndSoftensCorner()
    {
        using var mask = ImageExtractor.CreateRoundedMask(100, 80, 18);
        Assert.Equal(255, mask.At<byte>(40, 50));
        Assert.Equal((byte)0, mask.At<byte>(0, 0));
        Assert.True(mask.At<byte>(1, 20) > 0);
    }
}
