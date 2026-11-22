using Xunit;

using Inlay.Cli;

namespace Inlay.Tests;

public sealed class ArgumentsTests
{
    [Fact]
    public void ParsesSingleInputAndOptions()
    {
        var parsed = Arguments.Parse(["photo.jpg", "--preview", "--report", "--corners", "keep"]);
        Assert.Null(parsed.Error);
        Assert.NotNull(parsed.Options);
        Assert.Equal("photo.jpg", parsed.Options.Input);
        Assert.True(parsed.Options.Preview);
        Assert.True(parsed.Options.Report);
        Assert.True(parsed.Options.KeepCorners);
    }

    [Fact]
    public void ParsesManualRectangle()
    {
        var parsed = Arguments.Parse(["photo.jpg", "--rect", "26,592,869,867"]);
        Assert.Null(parsed.Error);
        Assert.Equal(new CropRectangle(26, 592, 869, 867), parsed.Options!.Rectangle);
    }

    [Theory]
    [InlineData("0,0,0,10")]
    [InlineData("-1,0,10,10")]
    [InlineData("1,2,3")]
    [InlineData("a,b,c,d")]
    public void RejectsInvalidRectangle(string value)
    {
        var parsed = Arguments.Parse(["photo.jpg", "--rect", value]);
        Assert.NotNull(parsed.Error);
    }

    [Fact]
    public void HelpIsShownWithoutArguments()
    {
        Assert.True(Arguments.Parse([]).ShowHelp);
    }
}
