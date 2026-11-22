using Inlay.Cli;
using Inlay.Processing;

var parsed = Arguments.Parse(args);
if (parsed.ShowHelp)
{
    Console.WriteLine(Arguments.HelpText);
    return 0;
}

if (parsed.Error is not null)
{
    Console.Error.WriteLine($"inlay: {parsed.Error}");
    Console.Error.WriteLine("Use 'inlay --help' for usage.");
    return 2;
}

try
{
    var summary = await new BatchProcessor().RunAsync(parsed.Options!);
    Console.WriteLine();
    Console.WriteLine($"Scanned {summary.Scanned}  |  Extracted {summary.Extracted}  |  Review {summary.Review}  |  Rejected {summary.Rejected}  |  Failed {summary.Failed}");
    return summary.Failed > 0 || summary.Rejected > 0 || summary.Review > 0 ? 1 : 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine($"inlay: {exception.Message}");
    return 2;
}
