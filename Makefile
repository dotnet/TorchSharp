all:
	msbuild /p:Configuration=Release

docs: doc-update yaml

doc-update:
	mdoc update -i ./TorchSharp/bin/Release/netstandard2.0/TorchSharp.xml -o ecmadocs/en ./TorchSharp/bin/Release/netstandard2.0/TorchSharp.dll

yaml:
	-rm ecmadocs/en/ns-.xml
	mono /cvs/ECMA2Yaml/ECMA2Yaml/ECMA2Yaml/bin/Debug/ECMA2Yaml.exe --source=`pwd`/ecmadocs/en --output=`pwd`/docfx/api
	(cd docfx; mono ~/Downloads/docfx/docfx.exe build)
