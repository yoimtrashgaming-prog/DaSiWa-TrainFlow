package trainer

import (
	"image"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Caption files an image can have. ".txt" is the primary caption (booru tags from
// the tagger, or whatever the user wrote). ".caption" is an optional second
// caption, normally natural language. An image with both trains once with each.
const (
	captionExtPrimary   = ".txt"
	captionExtSecondary = ".caption"
)

// datasetFolder is one folder of training images inside the dataset path. The
// sd-scripts profiles train every folder that holds images, so a dataset can be
// sorted into subfolders.
type datasetFolder struct {
	Path     string
	Rel      string // relative to the dataset path, "." for the top folder
	Images   []string
	Captions map[string]int // caption extension -> images in this folder that have one
}

// hasAll reports whether every image in the folder has a caption with ext.
func (f datasetFolder) hasAll(ext string) bool {
	return len(f.Images) > 0 && f.Captions[ext] == len(f.Images)
}

// scanImageDataset walks the dataset path and returns every folder that holds
// images, top folder first, then subfolders in path order. It skips hidden
// folders and the top-level "input" folder, where Dataset Prep moves the
// originals, so they never train alongside the prepared copies.
func scanImageDataset(datasetPath string) []datasetFolder {
	root := filepath.Clean(datasetPath)
	var folders []datasetFolder
	_ = filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil || !entry.IsDir() {
			return nil
		}
		if path != root && skipDatasetDir(root, path, entry.Name()) {
			return filepath.SkipDir
		}
		if folder, ok := readDatasetFolder(root, path); ok {
			folders = append(folders, folder)
		}
		return nil
	})
	sort.SliceStable(folders, func(i, j int) bool {
		if folders[i].Rel == "." || folders[j].Rel == "." {
			return folders[i].Rel == "."
		}
		return folders[i].Rel < folders[j].Rel
	})
	return folders
}

func skipDatasetDir(root, path, name string) bool {
	if strings.HasPrefix(name, ".") || name == "__pycache__" {
		return true
	}
	return path == preparedDatasetInputPath(Settings{DatasetPath: root})
}

func readDatasetFolder(root, dir string) (datasetFolder, bool) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return datasetFolder{}, false
	}
	files := map[string]bool{}
	for _, entry := range entries {
		if !entry.IsDir() {
			files[entry.Name()] = true
		}
	}
	rel, _ := filepath.Rel(root, dir)
	folder := datasetFolder{Path: dir, Rel: filepath.ToSlash(rel), Captions: map[string]int{}}
	for _, entry := range entries {
		if entry.IsDir() || !validImageExt(entry.Name()) {
			continue
		}
		folder.Images = append(folder.Images, entry.Name())
		stem := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
		for _, ext := range []string{captionExtPrimary, captionExtSecondary} {
			if files[stem+ext] {
				folder.Captions[ext]++
			}
		}
	}
	return folder, len(folder.Images) > 0
}

// captionSets returns the caption extensions to train and, for each, the folders
// that train with it. ".txt" always trains every folder. ".caption" trains only
// the folders where every image has one; validateSettings rejects a folder that
// has them for some images and not others.
func captionSets(folders []datasetFolder) map[string][]datasetFolder {
	sets := map[string][]datasetFolder{captionExtPrimary: folders}
	for _, folder := range folders {
		if folder.hasAll(captionExtSecondary) {
			sets[captionExtSecondary] = append(sets[captionExtSecondary], folder)
		}
	}
	return sets
}

// hasSecondaryCaptions reports whether the dataset trains a second caption per
// image. Text encoder outputs then have to be cached in memory: the disk cache
// is one file per image and cannot hold two captions.
func hasSecondaryCaptions(datasetPath string) bool {
	return len(captionSets(scanImageDataset(datasetPath))[captionExtSecondary]) > 0
}

// countDatasetImagesRecursive counts images in every training folder.
func countDatasetImagesRecursive(datasetPath string) int {
	count := 0
	for _, folder := range scanImageDataset(datasetPath) {
		count += len(folder.Images)
	}
	return count
}

func imageConfig(path string) (image.Config, bool) {
	file, err := os.Open(path)
	if err != nil {
		return image.Config{}, false
	}
	defer file.Close()
	cfg, _, err := image.DecodeConfig(file)
	return cfg, err == nil
}
