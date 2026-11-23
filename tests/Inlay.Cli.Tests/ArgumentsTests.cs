using Inlay.Cli;
using Inlay.Processing;
using Xunit;

namespace Inlay.Cli.Tests;

public sealed class ArgumentsTests
{
    [Fact]
    public void ParsesSingleInputAndOptions()
    {
        var result = Arguments.CreateCommand().Parse(["photo.jpg", "--preview", "--report", "--corners", "keep"]);

        Assert.Empty(result.Errors);
    }

    [Fact]
    public void ParsesBatchOptions()
    {
        var result = Arguments.CreateCommand().Parse(["photos", "--recursive", "--out", "extracted", "--overwrite"]);

        Assert.Empty(result.Errors);
    }

    [Fact]
    public void RejectsUnknownOptions()
    {
        var result = Arguments.CreateCommand().Parse(["photo.jpg", "--unknown"]);

        Assert.NotEmpty(result.Errors);
    }

    [Theory]
    [InlineData("26,592,869,867", 26, 592, 869, 867)]
    public void ParsesManualRectangle(string value, int x, int y, int width, int height)
    {
        Assert.True(Arguments.TryParseRectangle(value, out var rectangle));
        Assert.Equal(new CropRectangle(x, y, width, height), rectangle);
    }

    [Theory]
    [InlineData("0,0,0,10")]
    [InlineData("-1,0,10,10")]
    [InlineData("1,2,3")]
    [InlineData("a,b,c,d")]
    public void RejectsInvalidRectangle(string value)
    {
        Assert.False(Arguments.TryParseRectangle(value, out _));
    }
}
