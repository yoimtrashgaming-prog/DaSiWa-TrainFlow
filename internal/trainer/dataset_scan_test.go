package trainer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeDataset creates empty files; the scanner and TOML writers only look at
// names, so the images do not need to decode.
func writeDataset(t *testing.T, root string, files ...string) {
	t.Helper()
	for _, name := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("x"), 0644); err != nil {
			t.Fatal(err)
		}
	}
}

func datasetTOMLFor(t *testing.T, dataset string, arch string) string {
	t.Helper()
	s := normalizeSettings(Settings{
		Architecture:              arch,
		ProjectName:               "scan",
		OutputPath:                t.TempDir(),
		DatasetPath:               dataset,
		TrainingSteps:             900,
		TrainBatchSize:            1,
		GradientAccumulationSteps: 1,
	})
	path, err := createDatasetTOML(s.ProjectName, s, profileFor(s), 1024, 1024, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestScanImageDataset_nestedSkipsInputAndHidden(t *testing.T) {
	root := t.TempDir()
	writeDataset(t, root,
		"1.png", "1.txt",
		"closeups/1.png", "closeups/1.txt",
		"closeups/outfits/a.jpg", "closeups/outfits/a.txt",
		"input/original.png", "input/original.txt",
		".cache/x.png",
		"empty/readme.md",
	)
	var rels []string
	for _, folder := range scanImageDataset(root) {
		rels = append(rels, folder.Rel)
	}
	if got, want := strings.Join(rels, ","), ".,closeups,closeups/outfits"; got != want {
		t.Errorf("folders = %q, want %q", got, want)
	}
	if got := countDatasetImagesRecursive(root); got != 3 {
		t.Errorf("recursive image count = %d, want 3", got)
	}
	if got := countDatasetImages(root); got != 1 {
		t.Errorf("top-level count (Musubi) = %d, want 1", got)
	}
}

func TestCreateDatasetTOML_flatDatasetUnchanged(t *testing.T) {
	// A flat .txt-only dataset must write exactly the historical TOML.
	root := t.TempDir()
	writeDataset(t, root, "1.png", "1.txt", "2.png", "2.txt")
	want := "[general]\nenable_bucket = true\nmin_bucket_reso = 256\nmax_bucket_reso = 1024\nbucket_reso_steps = 32\nbucket_no_upscale = true\n\n" +
		"[[datasets]]\nresolution = 1024\n\n[[datasets.subsets]]\nimage_dir = " + tomlString(filepath.ToSlash(absPath(root))) +
		"\ncaption_extension = \".txt\"\nnum_repeats = 450\ncaption_prefix = \"\"\nkeep_tokens = 1\n"
	if got := datasetTOMLFor(t, root, ArchitectureSDXL); got != want {
		t.Errorf("flat dataset TOML changed:\n got: %q\nwant: %q", got, want)
	}
}

func TestCreateDatasetTOML_subfoldersAndSecondCaption(t *testing.T) {
	root := t.TempDir()
	writeDataset(t, root,
		"1.png", "1.txt", "1.caption",
		"2.png", "2.txt", "2.caption",
		"poses/1.png", "poses/1.txt",
	)
	toml := datasetTOMLFor(t, root, ArchitectureSDXL)
	if n := strings.Count(toml, "[[datasets]]\n"); n != 2 {
		t.Errorf("want one [[datasets]] per caption type (2), got %d:\n%s", n, toml)
	}
	if n := strings.Count(toml, "[[datasets.subsets]]"); n != 3 {
		t.Errorf("want 3 subsets (.txt: top + poses, .caption: top), got %d:\n%s", n, toml)
	}
	if !strings.Contains(toml, "image_dir = "+tomlString(filepath.ToSlash(absPath(filepath.Join(root, "poses"))))) {
		t.Errorf("subfolder missing:\n%s", toml)
	}
	captionBlock := toml[strings.LastIndex(toml, "[[datasets]]"):]
	if !strings.Contains(captionBlock, "caption_extension = \".caption\"") || strings.Contains(captionBlock, "poses") {
		t.Errorf(".caption block should hold only the fully captioned top folder:\n%s", captionBlock)
	}
	// 3 .txt entries + 2 .caption entries = 5; ceil(900/5) = 180.
	if !strings.Contains(toml, "num_repeats = 180\n") {
		t.Errorf("repeats should count both caption types:\n%s", toml)
	}
}

func TestCreateTrainingTOML_secondCaptionKeepsTextEncoderCacheInMemory(t *testing.T) {
	for _, tc := range []struct {
		name     string
		files    []string
		wantDisk bool
	}{
		{"txt only", []string{"1.png", "1.txt"}, true},
		{"txt and caption", []string{"1.png", "1.txt", "1.caption"}, false},
	} {
		for _, arch := range []string{ArchitectureAnima, ArchitectureSDXL} {
			root := t.TempDir()
			writeDataset(t, root, tc.files...)
			s := normalizeSettings(Settings{
				Architecture:  arch,
				ProjectName:   "cache",
				OutputPath:    t.TempDir(),
				DatasetPath:   root,
				NetworkRank:   16,
				NetworkAlpha:  16,
				LearningRate:  "1e-4",
				TrainingSteps: 100,
				SaveSteps:     100,
				TrainUNetOnly: true,
			})
			path, err := createTrainingTOML(s.ProjectName, s, profileFor(s), s.OutputPath, "", t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			data, _ := os.ReadFile(path)
			toml := string(data)
			if !strings.Contains(toml, "cache_text_encoder_outputs = true") {
				t.Errorf("%s/%s: text encoder outputs should still be cached", arch, tc.name)
			}
			if got := strings.Contains(toml, "cache_text_encoder_outputs_to_disk = true"); got != tc.wantDisk {
				t.Errorf("%s/%s: disk cache = %t, want %t", arch, tc.name, got, tc.wantDisk)
			}
		}
	}
}

func TestDatasetTagCommands_oneFolderAtATimeSkippingInput(t *testing.T) {
	root := t.TempDir()
	writeDataset(t, root, "1.png", "poses/1.png", "input/original.png")
	commands := datasetTagCommands("/tf", Settings{DatasetPath: root})
	var dirs []string
	for _, command := range commands {
		dirs = append(dirs, command[1])
		for _, arg := range command {
			if arg == "--recursive" {
				t.Errorf("tagger must not recurse into input/: %v", command)
			}
		}
	}
	want := []string{filepath.ToSlash(absPath(root)), filepath.ToSlash(absPath(filepath.Join(root, "poses")))}
	if strings.Join(dirs, "|") != strings.Join(want, "|") {
		t.Errorf("tagged folders = %v, want %v", dirs, want)
	}
}

func TestValidateSettings_partialSecondCaption(t *testing.T) {
	root := t.TempDir()
	writeDataset(t, root,
		"1.png", "1.txt", "1.caption",
		"2.png", "2.txt",
		"sub/1.png",
	)
	errs := strings.Join(validateSettings(Settings{Architecture: ArchitectureSDXL, DatasetPath: root}), "\n")
	if !strings.Contains(errs, `Folder ".": 1 of 2 images have a .caption file`) {
		t.Errorf("want a partial .caption error for the top folder, got:\n%s", errs)
	}
	if !strings.Contains(errs, "1 images are missing .txt captions") {
		t.Errorf("want the subfolder image counted as missing a .txt, got:\n%s", errs)
	}
}
