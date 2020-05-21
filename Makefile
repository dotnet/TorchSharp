all:
	sh build.sh

docs: doc-update yaml

doc-update:
	 mdoc update -i ./bin/obj/packprep/Debug/TorchSharp/lib/netstandard2.0/TorchSharp.xml -o ecmadocs/en ./bin/obj/packprep/Debug/TorchSharp/lib/netstandard2.0/TorchSharp.dll

yaml:
	-rm ecmadocs/en/ns-.xml
	mono /cvs/ECMA2Yaml/ECMA2Yaml/ECMA2Yaml/bin/Debug/ECMA2Yaml.exe --source=`pwd`/ecmadocs/en --output=`pwd`/docfx/api
	(cd docfx; mono ~/Downloads/docfx/docfx.exe build)
