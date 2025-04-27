package count

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"io"
	"math"
	"math/rand"
	"unsafe"

	"github.com/apache/datasketches-go/internal"
)

type countMinSketch[T any] struct {
	numBuckets  int32 // counter array size for each of the hashing function
	numHashes   int8  // number of hashing functions
	sketchSlice []int64
	seed        int64
	totalWeight int64
	hashSeeds   []int64
}

func NewCountMinSketch[T any](numHashes int8, numBuckets int32, seed int64) (*countMinSketch[T], error) {
	if numBuckets < 3 {
		return nil, errors.New("using fewer than 3 buckets incurs relative error greater than 1.0")
	}

	if numBuckets*int32(numHashes) >= 1<<30 {
		return nil, errors.New("these parameters generate a sketch that exceeds 2^30 elements")
	}

	rng := rand.New(rand.NewSource(seed))
	hashSeeds := make([]int64, numHashes)
	for i := range int(numHashes) {
		hashSeeds[i] = int64(rng.Int()) + seed
	}

	sketchSize := int(numBuckets * int32(numHashes))
	sketchSlice := make([]int64, sketchSize)

	return &countMinSketch[T]{
		numBuckets:  numBuckets,
		numHashes:   numHashes,
		sketchSlice: sketchSlice,
		seed:        seed,
		hashSeeds:   hashSeeds,
	}, nil
}

func (c *countMinSketch[T]) getNumBuckets() int32 {
	return c.numBuckets
}

func (c *countMinSketch[T]) getNumHashes() int8 {
	return c.numHashes
}

func (c *countMinSketch[T]) getTotalWeight() int64 {
	return c.totalWeight
}

func (c *countMinSketch[T]) getSeed() int64 {
	return c.seed
}

func (c *countMinSketch[T]) getRelativeError() float64 {
	return math.Exp(1.0) / float64(c.numBuckets)
}

func (c *countMinSketch[T]) isEmpty() bool {
	return c.totalWeight == 0
}

func (c *countMinSketch[T]) getHashes(item []byte) []int64 {
	var bucketIndex, hashSeedIndex uint64
	sketchUpdateLocations := make([]int64, c.numHashes)

	for i, s := range c.hashSeeds {
		h1, _ := internal.HashByteArrMurmur3(item, 0, len(item), uint64(s))
		bucketIndex = h1 % uint64(c.numBuckets)
		sketchUpdateLocations[i] = int64(hashSeedIndex)*int64(c.numBuckets) + int64(bucketIndex)
		hashSeedIndex++
	}

	return sketchUpdateLocations
}

func (c *countMinSketch[T]) update(item []byte, weight int64) error {
	if len(item) == 0 {
		return nil
	}

	if weight < 0 {
		c.totalWeight += -weight
	} else {
		c.totalWeight += weight
	}

	hashLocations := c.getHashes(item)
	for _, h := range hashLocations {
		c.sketchSlice[h] += weight
	}
	return nil
}

func (c *countMinSketch[T]) Update(item T, weight int64) error {
	if unsafe.Sizeof(item) == 0 {
		return nil
	}

	var b []byte
	buf := bytes.NewBuffer(b)
	encoder := gob.NewEncoder(buf)
	err := encoder.Encode(item)
	if err != nil {
		return err
	}

	return c.update(buf.Bytes(), weight)
}

func (c *countMinSketch[T]) getEstimate(item []byte) int64 {
	if len(item) == 0 {
		return 0
	}

	hashLocations := c.getHashes(item)
	estimate := int64(math.MaxInt64)
	for _, h := range hashLocations {
		estimate = Min(estimate, c.sketchSlice[h])
	}
	return estimate
}

func (c *countMinSketch[T]) GetEstimate(item T) int64 {
	if unsafe.Sizeof(item) == 0 {
		return 0
	}

	var b []byte
	buf := bytes.NewBuffer(b)
	encoder := gob.NewEncoder(buf)
	err := encoder.Encode(item)
	if err != nil {
		return -1
	}
	return c.getEstimate(buf.Bytes())
}

func (c *countMinSketch[T]) GetUpperBound(item T) int64 {
	return c.GetEstimate(item) + int64(c.getRelativeError()*float64(c.getTotalWeight()))
}

func (c *countMinSketch[T]) GetLowerBound(item T) int64 {
	return c.GetEstimate(item)
}

func (c *countMinSketch[T]) Merge(otherSketch *countMinSketch[T]) error {
	if c == otherSketch {
		return errors.New("cannot merge sketch with itself")
	}

	canMerge := c.getNumHashes() == otherSketch.getNumHashes() &&
		c.getNumBuckets() == otherSketch.getNumBuckets() &&
		c.getSeed() == otherSketch.getSeed()

	if !canMerge {
		return errors.New("sketches are incompatible")
	}

	for i := range c.sketchSlice {
		c.sketchSlice[i] += otherSketch.sketchSlice[i]
	}
	c.totalWeight += otherSketch.totalWeight

	return nil
}

func (c *countMinSketch[T]) Serialize(w io.Writer) error {
	preambleLongs := byte(PREAMBLE_LONGS_SHORT)
	serVer := byte(SERIAL_VERSION_1)
	familyID := byte(FAMILY_ID)

	var flagsByte byte
	if c.isEmpty() {
		flagsByte |= 1 << IS_EMPTY
	}
	unused32 := uint32(NULL_32)

	if err := binary.Write(w, binary.LittleEndian, preambleLongs); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, serVer); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, familyID); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, flagsByte); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, unused32); err != nil {
		return err
	}

	seedHash, err := internal.ComputeSeedHash(c.seed)
	if err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, c.numBuckets); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, c.numHashes); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, seedHash); err != nil {
		return err
	}
	unused8 := byte(NULL_8)
	if err := binary.Write(w, binary.LittleEndian, unused8); err != nil {
		return err
	}

	// Skip rest if sketch is empty
	if c.isEmpty() {
		return nil
	}

	if err := binary.Write(w, binary.LittleEndian, c.totalWeight); err != nil {
		return err
	}

	for _, h := range c.sketchSlice {
		err := binary.Write(w, binary.LittleEndian, h)
		if err != nil {
			return err
		}
	}

	return nil
}

func (c *countMinSketch[T]) deserialize(b []byte, seed int64) (*countMinSketch[T], error) {
	r := bytes.NewReader(b)
	preamble, err := r.ReadByte()
	if err != nil {
		return nil, err
	}

	serVe, err := r.ReadByte()
	if err != nil {
		return nil, err
	}

	familyID, err := r.ReadByte()
	if err != nil {
		return nil, err
	}

	flag, err := r.ReadByte()
	if err != nil {
		return nil, err
	}

	err = checkHeaderValidity(preamble, serVe, familyID, flag)
	if err != nil {
		return nil, err
	}

	unused32 := make([]byte, 4)
	_, err = r.Read(unused32)
	if err != nil {
		return nil, err
	}

	var numBuckets int32
	err = binary.Read(r, binary.LittleEndian, &numBuckets)
	if err != nil {
		return nil, err
	}

	var numHashes int8
	err = binary.Read(r, binary.LittleEndian, &numHashes)
	if err != nil {
		return nil, err
	}

	var seedHash int16
	err = binary.Read(r, binary.LittleEndian, &seedHash)
	if err != nil {
		return nil, err
	}

	var unused8 int8
	err = binary.Read(r, binary.LittleEndian, &unused8)
	if err != nil {
		return nil, err
	}

	cms, err := NewCountMinSketch[T](numHashes, numBuckets, seed)
	if err != nil {
		return nil, err
	}

	isEmpty := (flag & (1 << IS_EMPTY)) > 0
	if isEmpty {
		return cms, nil
	}

	var totalWeight int64
	err = binary.Read(r, binary.LittleEndian, &totalWeight)
	if err != nil {
		return nil, err
	}
	cms.totalWeight = totalWeight

	var w int64
	var i int
	for r.Len() > 0 {
		err = binary.Read(r, binary.LittleEndian, &w)
		if err != nil {
			return nil, err
		}
		cms.sketchSlice[i] = w
		i++
	}

	return cms, nil
}
