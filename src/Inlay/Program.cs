using Inlay.Cli;

var command = Arguments.CreateCommand();
return command.Parse(args.Length == 0 ? ["--help"] : args).Invoke();
