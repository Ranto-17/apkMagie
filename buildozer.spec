[app]
title = Galerie Photo
package.name = galeriephoto
package.domain = org.magic
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
requirements = python3,kivy,numpy,opencv,pillow
orientation = portrait
fullscreen = 1
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.archs = arm64-v8a
android.allow_backup = True

[buildozer]
log_level = 2
warn_on_root = 1
