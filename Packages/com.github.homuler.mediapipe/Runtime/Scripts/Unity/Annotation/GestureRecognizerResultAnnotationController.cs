// Copyright (c) 2023 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

using System.Collections.Generic;
using UnityEngine;

using Mediapipe.Tasks.Vision.GestureRecognizer;
using Mediapipe.Tasks.Components.Containers;

namespace Mediapipe.Unity
{
  public class GestureRecognizerResultAnnotationController : AnnotationController<MultiHandLandmarkListAnnotation>
  {
    [SerializeField] private bool _visualizeZ = false;
    [SerializeField] private LabelAnnotation _labelPrefab;

    private readonly object _currentTargetLock = new object();
    private GestureRecognizerResult _currentTarget;
    private readonly List<LabelAnnotation> _labels = new List<LabelAnnotation>();

    public void DrawNow(GestureRecognizerResult target)
    {
      target.CloneTo(ref _currentTarget);
      SyncNow();
    }

    public void DrawLater(GestureRecognizerResult target) => UpdateCurrentTarget(target);

    protected void UpdateCurrentTarget(GestureRecognizerResult newTarget)
    {
      lock (_currentTargetLock)
      {
        newTarget.CloneTo(ref _currentTarget);
        isStale = true;
      }
    }

    protected override void SyncNow()
    {
      lock (_currentTargetLock)
      {
        isStale = false;
        annotation.SetHandedness(_currentTarget.handedness);
        annotation.Draw(_currentTarget.handLandmarks, _visualizeZ);

        EnsureLabelCount(_currentTarget.gestures.Count);
        for (int i = 0; i < _labels.Count; i++)
        {
          if (i < _currentTarget.gestures.Count &&
              _currentTarget.gestures[i].categories.Count > 0 &&
              i < _currentTarget.handLandmarks.Count)
          {
            var category = _currentTarget.gestures[i].categories[0];
            var landmark = _currentTarget.handLandmarks[i][0];
            var pos = new Vector3(landmark.x * imageSize.x, landmark.y * imageSize.y, 0);
            _labels[i].Draw(category.categoryName, pos, Color.green, imageSize.x, imageSize.y);
          }
          else
          {
            _labels[i].Draw(null, Vector3.zero, Color.clear, 0, 0);
          }
        }
      }
    }

    private void EnsureLabelCount(int count)
    {
      while (_labels.Count < count)
      {
        _labels.Add(Instantiate(_labelPrefab, transform));
      }
    }
  }
}
