; Inno Setup script: dist\SongPolish (PyInstaller folder) -> dist\SongPolish-<version>-windows-setup.exe
; Build: iscc /DVersion=0.1.0 packaging\windows\installer.iss
; Per-user install (no administrator rights), Start menu entry, optional desktop icon, uninstaller.

#ifndef Version
  #define Version "0.0.0"
#endif

[Setup]
AppId={{6F3B8E2C-5A41-4C0B-9E7D-2B9A1F4C8D13}
AppName=SongPolish
AppVersion={#Version}
AppPublisher=Jose J. Rodriguez
DefaultDirName={localappdata}\Programs\SongPolish
DefaultGroupName=SongPolish
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=SongPolish-{#Version}-windows-setup
SetupIconFile=songpolish.ico
UninstallDisplayIcon={app}\SongPolish.exe
Compression=lzma2/fast
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\dist\SongPolish\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SongPolish"; Filename: "{app}\SongPolish.exe"
Name: "{group}\Uninstall SongPolish"; Filename: "{uninstallexe}"
Name: "{autodesktop}\SongPolish"; Filename: "{app}\SongPolish.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SongPolish.exe"; Description: "{cm:LaunchProgram,SongPolish}"; Flags: nowait postinstall skipifsilent
